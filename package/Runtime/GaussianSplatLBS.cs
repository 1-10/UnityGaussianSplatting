// SPDX-License-Identifier: MIT

using System;
using Unity.Mathematics;
using UnityEngine;

namespace GaussianSplatting.Runtime
{
    // Applies Linear Blend Skinning (LBS) to a GaussianSplatRenderer's splat positions every frame,
    // driven by a FLAME-style jaw_pose (axis-angle) value. Per-splat bone weights are loaded from a
    // JSON file (see LBSWeightsData) at Start, and the resulting bone matrices are used to deform the
    // splat positions on the GPU via a compute shader dispatch in LateUpdate.
    [ExecuteInEditMode]
    public class GaussianSplatLBS : MonoBehaviour
    {
        // Bone layout expected in the LBS weight data and bone matrix buffer.
        public enum BoneIndex
        {
            Root = 0,
            Neck = 1,
            Jaw = 2,
            LeftEye = 3,
            RightEye = 4,
        }
        public const int kBoneCount = 5;

        [Tooltip("GaussianSplatRenderer whose splat positions will be skinned. If not set, will use the one on this GameObject.")]
        [SerializeField] GaussianSplatRenderer m_Renderer;
        [Tooltip("JSON file containing per-splat bone weights (lbs_weight_20k.json). If not set, will try to load a resource named 'lbs_weight_20k'.")]
        [SerializeField] TextAsset m_LBSWeightsJson;
        [Tooltip("Compute shader implementing the LBS kernel (GaussianSplatLBS.compute).")]
        [SerializeField] ComputeShader m_CSLbs;
        [Tooltip("Jaw opening angle (radians) used when SetJawPose has not been called externally.")]
        [SerializeField, Range(0f, 1.2f)] float m_JawAngle;

        GraphicsBuffer m_GpuLBSWeights;
        GraphicsBuffer m_GpuBoneMatrices;
        int m_WeightPointCount;
        int m_KernelApplyLBS = -1;
        float3 m_JawPose;
        bool m_JawPoseSetExternally;

        [Serializable]
        class LBSWeightsData
        {
            public int pointCount;
            public int boneCount;
            public float[] weights; // flattened, length == pointCount * boneCount
        }

        public GaussianSplatRenderer targetRenderer
        {
            get => m_Renderer;
            set => m_Renderer = value;
        }

        void Start()
        {
            if (m_Renderer == null)
                m_Renderer = GetComponent<GaussianSplatRenderer>();

            LoadLbsWeights();

            if (m_CSLbs != null)
                m_KernelApplyLBS = m_CSLbs.FindKernel("CSApplyLBS");

            InitBoneMatricesBuffer();
        }

        void LoadLbsWeights()
        {
            var json = m_LBSWeightsJson;
            if (json == null)
                json = Resources.Load<TextAsset>("lbs_weight_20k");
            if (json == null)
            {
                Debug.LogWarning($"{nameof(GaussianSplatLBS)}: no LBS weight JSON assigned or found (lbs_weight_20k), skinning is disabled.");
                return;
            }

            LBSWeightsData data;
            try
            {
                data = JsonUtility.FromJson<LBSWeightsData>(json.text);
            }
            catch (Exception e)
            {
                Debug.LogError($"{nameof(GaussianSplatLBS)}: failed to parse LBS weight JSON: {e.Message}");
                return;
            }

            if (data?.weights == null || data.boneCount <= 0 || data.pointCount <= 0 ||
                data.weights.Length != data.pointCount * data.boneCount)
            {
                Debug.LogError($"{nameof(GaussianSplatLBS)}: LBS weight JSON is malformed.");
                return;
            }

            m_WeightPointCount = data.pointCount;
            m_GpuLBSWeights?.Dispose();
            m_GpuLBSWeights = new GraphicsBuffer(GraphicsBuffer.Target.Structured, data.weights.Length, sizeof(float))
                { name = "GaussianLBSWeights" };
            m_GpuLBSWeights.SetData(data.weights);
        }

        void InitBoneMatricesBuffer()
        {
            m_GpuBoneMatrices?.Dispose();
            m_GpuBoneMatrices = new GraphicsBuffer(GraphicsBuffer.Target.Structured, kBoneCount, 16 * sizeof(float))
                { name = "GaussianLBSBoneMatrices" };
            var identity = new float4x4[kBoneCount];
            for (int i = 0; i < kBoneCount; i++)
                identity[i] = float4x4.identity;
            m_GpuBoneMatrices.SetData(identity);
        }

        // Sets the current jaw_pose (axis-angle, radians) to drive the jaw bone. Once called, this
        // takes precedence over the Inspector jawAngle slider.
        public void SetJawPose(float3 axisAngle)
        {
            m_JawPose = axisAngle;
            m_JawPoseSetExternally = true;
        }

        void LateUpdate()
        {
            if (m_Renderer == null || !m_Renderer.HasValidRenderSetup)
                return;
            if (m_CSLbs == null || m_KernelApplyLBS < 0)
                return;
            if (m_GpuLBSWeights == null || m_GpuBoneMatrices == null)
                return;
            if (m_Renderer.splatCount != m_WeightPointCount)
                return;

            // The Inspector slider rotates the jaw around the x axis, so the axis-angle vector's
            // magnitude is m_JawAngle and its direction is (1,0,0).
            float3 jawAxisAngle = m_JawPoseSetExternally ? m_JawPose : new float3(m_JawAngle, 0, 0);
            UpdateBoneMatrices(jawAxisAngle);
            DispatchLBS();
        }

        void UpdateBoneMatrices(float3 jawAxisAngle)
        {
            var matrices = new float4x4[kBoneCount];
            for (int i = 0; i < kBoneCount; i++)
                matrices[i] = float4x4.identity;

            float angle = math.length(jawAxisAngle);
            quaternion jawRot = angle > 1.0e-6f
                ? quaternion.AxisAngle(jawAxisAngle / angle, angle)
                : quaternion.identity;
            matrices[(int)BoneIndex.Jaw] = new float4x4(jawRot, float3.zero);

            m_GpuBoneMatrices.SetData(matrices);
        }

        void DispatchLBS()
        {
            var asset = m_Renderer.asset;
            if (asset == null)
                return;

            var cs = m_CSLbs;
            int kernel = m_KernelApplyLBS;
            int splatCount = m_Renderer.splatCount;

            cs.SetBuffer(kernel, "_SplatPos", m_Renderer.m_GpuPosData);
            cs.SetBuffer(kernel, "_SplatChunks", m_Renderer.m_GpuChunks);
            cs.SetBuffer(kernel, "_LBSWeights", m_GpuLBSWeights);
            cs.SetBuffer(kernel, "_BoneMatrices", m_GpuBoneMatrices);

            cs.SetInt("_SplatCount", splatCount);
            cs.SetInt("_SplatChunkCount", m_Renderer.m_GpuChunksValid ? m_Renderer.m_GpuChunks.count : 0);
            cs.SetInt("_SplatFormat", (int)asset.posFormat);
            cs.SetInt("_BoneCount", kBoneCount);

            cs.GetKernelThreadGroupSizes(kernel, out uint groupSizeX, out _, out _);
            int groups = (int)((splatCount + groupSizeX - 1) / groupSizeX);
            cs.Dispatch(kernel, groups, 1, 1);
        }

        void OnDestroy()
        {
            m_GpuLBSWeights?.Dispose();
            m_GpuLBSWeights = null;
            m_GpuBoneMatrices?.Dispose();
            m_GpuBoneMatrices = null;
        }
    }
}
