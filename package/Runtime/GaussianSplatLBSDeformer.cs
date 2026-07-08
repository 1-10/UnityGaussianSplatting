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
    // Requires the target GaussianSplatRenderer to use VectorFormat.Norm11 or
    // VectorFormat.Float32 splat positions. Norm11 (the default format for
    // GaussianSplatAsset) re-normalizes deformed positions against the
    // canonical mesh's per-chunk bounding box and clamps them with a
    // saturate(), so any deformation that moves a splat outside that box will
    // visibly clamp/stick at the boundary. VectorFormat.Float32 is
    // recommended for assets driven by this component (larger asset size is
    // an acceptable tradeoff for per-avatar head splats), since deformed
    // positions are written back unclamped.
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

        [Tooltip("Binary asset (.bytes) containing the FLAME expression blend-shape basis. " +
                 "Layout: splatCount * exprDim * 3 contiguous IEEE-754 little-endian floats " +
                 "(one float3 per splat per expression dimension, row-major). " +
                 "exprDim is derived at runtime from the asset size. Leave null to disable expression deformation.")]
        public TextAsset m_ExprBasis;

        [Tooltip("FLAME expression coefficients (exprDim floats). Set from external code every frame. " +
                 "Ignored when m_ExprBasis is null.")]
        public float[] m_ExprCoeffs;

        GaussianSplatRenderer m_Splat;
        GraphicsBuffer m_GpuBoneWeights;
        GraphicsBuffer m_GpuBoneMatrices;
        // Canonical (un-deformed) splat positions, captured once so that every
        // frame's LBS pass reads from the original pose instead of the
        // previous frame's already-deformed m_GpuPosData (which would
        // otherwise accumulate deformation across frames).
        GraphicsBuffer m_GpuPosCanonical;
        int m_KernelIndex = -1;
        int m_WeightSplatCount;
        readonly float4x4[] m_BoneMatricesScratch = new float4x4[kBoneCount];

        // GPU resources for FLAME expression blend shapes.
        GraphicsBuffer m_GpuExprBasis;    // splatCount * exprDim * 3 floats
        GraphicsBuffer m_GpuExprCoeffs;   // exprDim floats
        GraphicsBuffer m_GpuPosWithExpr;  // splatCount float3 values (decoded, expr-applied)
        int m_ExprKernelApply = -1;       // CSApplyExpr kernel index
        int m_ExprKernelLBS = -1;         // CSApplyLBSWithExpr kernel index
        int m_ExprDim;                    // number of expression dimensions (K)
        int m_ExprSplatCount;             // splatCount when expr buffers were last built
        TextAsset m_ExprBasisAsset;       // m_ExprBasis at the time expr buffers were last built
        bool m_ExprCoeffLengthWarned;     // suppresses per-frame log spam for coefficient length mismatch

        static class Props
        {
            public static readonly int SplatPos = Shader.PropertyToID("_SplatPos");
            public static readonly int SplatPosCanonical = Shader.PropertyToID("_SplatPosCanonical");
            public static readonly int SplatChunks = Shader.PropertyToID("_SplatChunks");
            public static readonly int SplatChunkCount = Shader.PropertyToID("_SplatChunkCount");
            public static readonly int SplatFormat = Shader.PropertyToID("_SplatFormat");
            public static readonly int SplatCount = Shader.PropertyToID("_SplatCount");
            public static readonly int BoneWeights = Shader.PropertyToID("_BoneWeights");
            public static readonly int BoneMatrices = Shader.PropertyToID("_BoneMatrices");
            public static readonly int ExprBasis = Shader.PropertyToID("_ExprBasis");
            public static readonly int ExprCoeffs = Shader.PropertyToID("_ExprCoeffs");
            public static readonly int ExprDim = Shader.PropertyToID("_ExprDim");
            public static readonly int SplatPosWithExpr = Shader.PropertyToID("_SplatPosWithExpr");
        }

        [Serializable]
        class LBSWeightsJson
        {
            // Flattened splatCount * kBoneCount weights, row-major per splat.
            public float[] weights;
        }

        public bool HasValidSetup =>
            m_Splat != null && m_Splat.HasValidRenderSetup && m_CSLBSDeform != null &&
            m_GpuBoneWeights != null && m_GpuPosCanonical != null && m_WeightSplatCount == m_Splat.splatCount &&
            SystemInfo.supportsComputeShaders;

        // True when the expression blend-shape buffers are built and consistent with the current splat count.
        bool HasExprSetup =>
            m_GpuExprBasis != null && m_GpuPosWithExpr != null &&
            m_ExprSplatCount == m_Splat?.splatCount && m_ExprBasisAsset == m_ExprBasis;

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
            m_GpuPosCanonical?.Dispose();
            m_GpuPosCanonical = null;
            m_WeightSplatCount = 0;

            DisposeExprBuffers();
        }

        // Releases all expression-blend-shape GPU buffers and resets tracking state.
        void DisposeExprBuffers()
        {
            m_GpuExprBasis?.Dispose();
            m_GpuExprBasis = null;
            m_GpuExprCoeffs?.Dispose();
            m_GpuExprCoeffs = null;
            m_GpuPosWithExpr?.Dispose();
            m_GpuPosWithExpr = null;
            m_ExprSplatCount = 0;
            m_ExprDim = 0;
            m_ExprBasisAsset = null;
            m_ExprCoeffLengthWarned = false;
        }

        void EnsureBuffers()
        {
            if (m_Splat == null || !m_Splat.HasValidRenderSetup || m_LBSWeights == null)
                return;

            if (m_GpuBoneWeights != null && m_WeightSplatCount == m_Splat.splatCount)
            {
                // Bone weights are already up to date; still check expr in case it changed.
                EnsureExprBuffers();
                return;
            }

            GaussianSplatAsset.VectorFormat posFormat = m_Splat.asset.posFormat;
            if (posFormat != GaussianSplatAsset.VectorFormat.Norm11 && posFormat != GaussianSplatAsset.VectorFormat.Float32)
            {
                Debug.LogError($"GaussianSplatLBSDeformer: unsupported splat position format '{posFormat}'. " +
                                "Only Norm11 and Float32 are supported (Float32 is recommended to avoid " +
                                "deformation clamping at the canonical mesh's per-chunk bounds).", this);
                return;
            }

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

            m_GpuPosCanonical?.Dispose();
            m_GpuPosCanonical = new GraphicsBuffer(m_Splat.m_GpuPosData.target | GraphicsBuffer.Target.CopyDestination,
                m_Splat.m_GpuPosData.count, m_Splat.m_GpuPosData.stride) { name = "GaussianSplatLBSPosCanonical" };
            Graphics.CopyBuffer(m_Splat.m_GpuPosData, m_GpuPosCanonical);

            if (m_GpuBoneMatrices == null)
            {
                m_GpuBoneMatrices = new GraphicsBuffer(GraphicsBuffer.Target.Structured, kBoneCount, UnsafeUtility.SizeOf<float4x4>())
                    { name = "GaussianSplatLBSBoneMatrices" };
            }

            m_KernelIndex = m_CSLBSDeform.FindKernel("CSApplyLBS");

            EnsureExprBuffers();
        }

        // Validates and (re)builds the GPU buffers for expression blend shapes.
        // Called from EnsureBuffers(). Safe to call when m_ExprBasis is null
        // (disposes any stale expr buffers and returns immediately).
        void EnsureExprBuffers()
        {
            if (m_ExprBasis == null)
            {
                // Expr disabled — dispose stale buffers if any.
                if (m_GpuExprBasis != null)
                    DisposeExprBuffers();
                return;
            }

            if (m_GpuExprBasis != null &&
                m_ExprSplatCount == m_Splat.splatCount &&
                m_ExprBasisAsset == m_ExprBasis)
                return; // Already up to date.

            DisposeExprBuffers();

            byte[] bytes = m_ExprBasis.bytes;
            int splatCount = m_Splat.splatCount;
            // Expected layout: splatCount * exprDim * 3 IEEE-754 floats (4 bytes each).
            int totalFloats = bytes.Length / 4;
            if (bytes.Length % 4 != 0 || totalFloats % (splatCount * 3) != 0)
            {
                Debug.LogError($"GaussianSplatLBSDeformer: expr basis '{m_ExprBasis.name}' has " +
                               $"{bytes.Length} bytes, which is not divisible by splatCount*3*4 = " +
                               $"{splatCount * 3 * 4}. Expected splatCount*exprDim*3 floats.", this);
                return;
            }
            int exprDim = totalFloats / (splatCount * 3);

            float[] floatData = new float[totalFloats];
            System.Buffer.BlockCopy(bytes, 0, floatData, 0, bytes.Length);

            m_GpuExprBasis = new GraphicsBuffer(GraphicsBuffer.Target.Structured, totalFloats, UnsafeUtility.SizeOf<float>())
                { name = "GaussianSplatLBSExprBasis" };
            m_GpuExprBasis.SetData(floatData);

            m_GpuExprCoeffs = new GraphicsBuffer(GraphicsBuffer.Target.Structured, exprDim, UnsafeUtility.SizeOf<float>())
                { name = "GaussianSplatLBSExprCoeffs" };
            m_GpuExprCoeffs.SetData(new float[exprDim]); // Initialize to zeros.

            // Raw byte-addressable buffer: splatCount float3 values (3 floats × 4 bytes = 12 bytes each).
            m_GpuPosWithExpr = new GraphicsBuffer(GraphicsBuffer.Target.Raw, splatCount * 3, UnsafeUtility.SizeOf<float>())
                { name = "GaussianSplatLBSPosWithExpr" };

            m_ExprDim = exprDim;
            m_ExprSplatCount = splatCount;
            m_ExprBasisAsset = m_ExprBasis;
            m_ExprKernelApply = m_CSLBSDeform.FindKernel("CSApplyExpr");
            m_ExprKernelLBS = m_CSLBSDeform.FindKernel("CSApplyLBSWithExpr");
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
            uint format = (uint)m_Splat.asset.posFormat;
            int splatCount = m_Splat.splatCount;
            int chunkCount = m_Splat.m_GpuChunksValid ? m_Splat.m_GpuChunks.count : 0;

            bool useExpr = HasExprSetup;

            if (useExpr)
            {
                // Upload expression coefficients (caller is expected to update m_ExprCoeffs every frame).
                if (m_ExprCoeffs != null && m_ExprCoeffs.Length == m_ExprDim)
                {
                    m_GpuExprCoeffs.SetData(m_ExprCoeffs);
                    m_ExprCoeffLengthWarned = false; // Reset so a future mismatch is reported again.
                }
                else if (m_ExprCoeffs != null && m_ExprCoeffs.Length != m_ExprDim && !m_ExprCoeffLengthWarned)
                {
                    Debug.LogWarning($"GaussianSplatLBSDeformer: m_ExprCoeffs has {m_ExprCoeffs.Length} elements " +
                                     $"but exprDim is {m_ExprDim}. Expression coefficients will not be updated.", this);
                    m_ExprCoeffLengthWarned = true;
                }
                // If m_ExprCoeffs is null, keep the previously uploaded values (zeros on first use).

                // Pass 1 — CSApplyExpr: decode canonical positions, add expression offset,
                //           write decoded float3 results to _SplatPosWithExpr.
                cs.SetBuffer(m_ExprKernelApply, Props.SplatPosCanonical, m_GpuPosCanonical);
                cs.SetBuffer(m_ExprKernelApply, Props.SplatChunks, m_Splat.m_GpuChunks);
                cs.SetInt(Props.SplatChunkCount, chunkCount);
                cs.SetInt(Props.SplatFormat, (int)format);
                cs.SetInt(Props.SplatCount, splatCount);
                cs.SetInt(Props.ExprDim, m_ExprDim);
                cs.SetBuffer(m_ExprKernelApply, Props.ExprBasis, m_GpuExprBasis);
                cs.SetBuffer(m_ExprKernelApply, Props.ExprCoeffs, m_GpuExprCoeffs);
                cs.SetBuffer(m_ExprKernelApply, Props.SplatPosWithExpr, m_GpuPosWithExpr);
                cs.GetKernelThreadGroupSizes(m_ExprKernelApply, out uint exprGsX, out _, out _);
                cs.Dispatch(m_ExprKernelApply, (splatCount + (int)exprGsX - 1) / (int)exprGsX, 1, 1);

                // Pass 2 — CSApplyLBSWithExpr: apply bone LBS reading from the expression-intermediate buffer.
                cs.SetBuffer(m_ExprKernelLBS, Props.SplatPos, m_Splat.m_GpuPosData);
                cs.SetBuffer(m_ExprKernelLBS, Props.SplatPosWithExpr, m_GpuPosWithExpr);
                cs.SetBuffer(m_ExprKernelLBS, Props.SplatChunks, m_Splat.m_GpuChunks);
                cs.SetInt(Props.SplatChunkCount, chunkCount);
                cs.SetInt(Props.SplatFormat, (int)format);
                cs.SetInt(Props.SplatCount, splatCount);
                cs.SetBuffer(m_ExprKernelLBS, Props.BoneWeights, m_GpuBoneWeights);
                cs.SetBuffer(m_ExprKernelLBS, Props.BoneMatrices, m_GpuBoneMatrices);
                cs.GetKernelThreadGroupSizes(m_ExprKernelLBS, out uint lbsGsX, out _, out _);
                cs.Dispatch(m_ExprKernelLBS, (splatCount + (int)lbsGsX - 1) / (int)lbsGsX, 1, 1);
            }
            else
            {
                // Original path: CSApplyLBS reads directly from the canonical buffer.
                cs.SetBuffer(m_KernelIndex, Props.SplatPos, m_Splat.m_GpuPosData);
                cs.SetBuffer(m_KernelIndex, Props.SplatPosCanonical, m_GpuPosCanonical);
                cs.SetBuffer(m_KernelIndex, Props.SplatChunks, m_Splat.m_GpuChunks);
                cs.SetInt(Props.SplatChunkCount, chunkCount);
                cs.SetInt(Props.SplatFormat, (int)format);
                cs.SetInt(Props.SplatCount, splatCount);
                cs.SetBuffer(m_KernelIndex, Props.BoneWeights, m_GpuBoneWeights);
                cs.SetBuffer(m_KernelIndex, Props.BoneMatrices, m_GpuBoneMatrices);
                cs.GetKernelThreadGroupSizes(m_KernelIndex, out uint gsX, out _, out _);
                cs.Dispatch(m_KernelIndex, (splatCount + (int)gsX - 1) / (int)gsX, 1, 1);
            }
        }
    }
}
