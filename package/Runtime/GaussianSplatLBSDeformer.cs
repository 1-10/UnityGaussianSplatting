// SPDX-License-Identifier: MIT

using System;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEngine;

namespace GaussianSplatting.Runtime
{
    // Deforms a GaussianSplatRenderer's splat positions on the GPU using
    // Linear Blend Skinning (LBS), driven by a FLAME-style axis-angle
    // "jaw_pose" and a per-splat bone weight table (root/neck/jaw/leftEye/rightEye).
    //
    // Requires the target GaussianSplatRenderer to use VectorFormat.Norm11
    // splat positions (the default format for GaussianSplatAsset).
    [ExecuteInEditMode]
    [RequireComponent(typeof(GaussianSplatRenderer))]
    public class GaussianSplatLBSDeformer : MonoBehaviour
    {
        public const int kBoneCount = 5;

        public enum Bone
        {
            Root = 0,
            Neck = 1,
            Jaw = 2,
            LeftEye = 3,
            RightEye = 4,
        }

        [Tooltip("Compute shader implementing the CSApplyLBS kernel")]
        public ComputeShader m_CSLBSDeform;

        [Tooltip("JSON asset with a flattened 'weights' float array of splatCount * 5 bone weights")]
        public TextAsset m_LBSWeights;

        [Tooltip("Jaw joint position in splat object space, used as the pivot for jaw_pose rotation")]
        public Vector3 m_JawPivot;

        [Tooltip("FLAME jaw_pose, an axis-angle rotation (radians) applied around m_JawPivot")]
        public Vector3 m_JawPose;

        GaussianSplatRenderer m_Splat;
        GraphicsBuffer m_GpuBoneWeights;
        GraphicsBuffer m_GpuBoneMatrices;
        int m_KernelIndex = -1;
        int m_WeightSplatCount;
        readonly float4x4[] m_BoneMatricesScratch = new float4x4[kBoneCount];

        static class Props
        {
            public static readonly int SplatPos = Shader.PropertyToID("_SplatPos");
            public static readonly int SplatChunks = Shader.PropertyToID("_SplatChunks");
            public static readonly int SplatChunkCount = Shader.PropertyToID("_SplatChunkCount");
            public static readonly int SplatFormat = Shader.PropertyToID("_SplatFormat");
            public static readonly int SplatCount = Shader.PropertyToID("_SplatCount");
            public static readonly int BoneWeights = Shader.PropertyToID("_BoneWeights");
            public static readonly int BoneMatrices = Shader.PropertyToID("_BoneMatrices");
        }

        [Serializable]
        class LBSWeightsJson
        {
            // Flattened splatCount * kBoneCount weights, row-major per splat.
            public float[] weights;
        }

        public bool HasValidSetup =>
            m_Splat != null && m_Splat.HasValidRenderSetup && m_CSLBSDeform != null &&
            m_GpuBoneWeights != null && m_WeightSplatCount == m_Splat.splatCount &&
            SystemInfo.supportsComputeShaders;

        // Computes an object-space affine matrix for a bone, rotating by
        // an axis-angle vector (radians, magnitude = angle) around pivot.
        public static float4x4 AxisAngleToMatrix(Vector3 axisAngle, Vector3 pivot)
        {
            float3 pivotF3 = new float3(pivot.x, pivot.y, pivot.z);
            float angle = axisAngle.magnitude;
            quaternion rot = quaternion.identity;
            if (angle > 1e-8f)
            {
                Vector3 axis = axisAngle / angle;
                rot = quaternion.AxisAngle(new float3(axis.x, axis.y, axis.z), angle);
            }
            float4x4 r = float4x4.TRS(pivotF3, rot, new float3(1, 1, 1));
            float4x4 invPivot = float4x4.Translate(-pivotF3);
            return math.mul(r, invPivot);
        }

        void OnEnable()
        {
            m_Splat = GetComponent<GaussianSplatRenderer>();
        }

        void OnDisable()
        {
            m_GpuBoneWeights?.Dispose();
            m_GpuBoneWeights = null;
            m_GpuBoneMatrices?.Dispose();
            m_GpuBoneMatrices = null;
            m_WeightSplatCount = 0;
        }

        void EnsureBuffers()
        {
            if (m_Splat == null || !m_Splat.HasValidRenderSetup || m_LBSWeights == null)
                return;

            if (m_GpuBoneWeights != null && m_WeightSplatCount == m_Splat.splatCount)
                return;

            m_GpuBoneWeights?.Dispose();
            m_GpuBoneWeights = null;
            m_WeightSplatCount = 0;

            var json = JsonUtility.FromJson<LBSWeightsJson>(m_LBSWeights.text);
            int expected = m_Splat.splatCount * kBoneCount;
            if (json?.weights == null || json.weights.Length != expected)
            {
                Debug.LogError($"GaussianSplatLBSDeformer: expected {expected} bone weights " +
                                $"({m_Splat.splatCount} splats x {kBoneCount} bones), got " +
                                $"{json?.weights?.Length ?? 0} in '{m_LBSWeights.name}'", this);
                return;
            }

            m_GpuBoneWeights = new GraphicsBuffer(GraphicsBuffer.Target.Structured, expected, UnsafeUtility.SizeOf<float>())
                { name = "GaussianSplatLBSBoneWeights" };
            m_GpuBoneWeights.SetData(json.weights);
            m_WeightSplatCount = m_Splat.splatCount;

            if (m_GpuBoneMatrices == null)
            {
                m_GpuBoneMatrices = new GraphicsBuffer(GraphicsBuffer.Target.Structured, kBoneCount, UnsafeUtility.SizeOf<float4x4>())
                    { name = "GaussianSplatLBSBoneMatrices" };
            }

            m_KernelIndex = m_CSLBSDeform.FindKernel("CSApplyLBS");
        }

        // Bone matrices are all identity except for the jaw, which is driven
        // by m_JawPose. Override BuildBoneMatrices for more elaborate rigs.
        protected virtual void BuildBoneMatrices(float4x4[] matrices)
        {
            for (int i = 0; i < kBoneCount; ++i)
                matrices[i] = float4x4.identity;
            matrices[(int)Bone.Jaw] = AxisAngleToMatrix(m_JawPose, m_JawPivot);
        }

        void LateUpdate()
        {
            ApplyLBS();
        }

        public void ApplyLBS()
        {
            EnsureBuffers();
            if (!HasValidSetup)
                return;

            var matrices = m_BoneMatricesScratch;
            BuildBoneMatrices(matrices);
            m_GpuBoneMatrices.SetData(matrices);

            ComputeShader cs = m_CSLBSDeform;
            cs.SetBuffer(m_KernelIndex, Props.SplatPos, m_Splat.m_GpuPosData);
            cs.SetBuffer(m_KernelIndex, Props.SplatChunks, m_Splat.m_GpuChunks);
            cs.SetInt(Props.SplatChunkCount, m_Splat.m_GpuChunksValid ? m_Splat.m_GpuChunks.count : 0);
            uint format = (uint)m_Splat.asset.posFormat;
            cs.SetInt(Props.SplatFormat, (int)format);
            cs.SetInt(Props.SplatCount, m_Splat.splatCount);
            cs.SetBuffer(m_KernelIndex, Props.BoneWeights, m_GpuBoneWeights);
            cs.SetBuffer(m_KernelIndex, Props.BoneMatrices, m_GpuBoneMatrices);

            cs.GetKernelThreadGroupSizes(m_KernelIndex, out uint gsX, out _, out _);
            int groupSize = (int)gsX;
            int groups = (m_Splat.splatCount + groupSize - 1) / groupSize;
            cs.Dispatch(m_KernelIndex, groups, 1, 1);
        }
    }
}
