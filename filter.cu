#include <iostream>
#include <stdio.h>
#include <vector>
#include <algorithm>

#include <adiak.hpp>
#include <caliper/cali-manager.h>
#include <caliper/cali.h>

// Define constant memory for the filter (3x3 = 9 elements)
__constant__ float c_filter[9];

// Define block dimensions
#define BLOCK_W 16
#define BLOCK_H 16

/**
 * @brief Implementation that uses shared memory for faster memory access along with vectorized operations 
 * and constant memory for the filter 
 * @param a input image
 * @param b output image
 * @param nx image width
 * @param ny image length
 */
__global__ void filter_shared_vector(unsigned char *a, unsigned char *b, int nx, int ny) {
    // Shared memory size: Block width * 4 (uchar4) + 2 (halos) by Block height + 2 (halos)
    // Width: 16 * 4 + 2 = 66
    __shared__ unsigned char s_mem[BLOCK_H + 2][BLOCK_W * 4 + 2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Calculate global coordinates (x is in pixels, processing 4 pixels per thread)
    int ix = (blockIdx.x * blockDim.x + tx) * 4;
    int iy = blockIdx.y * blockDim.y + ty;

    // Determine 1D index for vectorized access (nx must be divisible by 4)
    int idx_vec = iy * (nx / 4) + (blockIdx.x * blockDim.x + tx);

    // 1. Load Main Data using Vectorized Load (uchar4)
    if (ix < nx && iy < ny) {
        uchar4 val = ((uchar4*)a)[idx_vec];
        // Unpack into shared memory, offset by 1 for halo
        s_mem[ty + 1][tx * 4 + 1] = val.x;
        s_mem[ty + 1][tx * 4 + 2] = val.y;
        s_mem[ty + 1][tx * 4 + 3] = val.z;
        s_mem[ty + 1][tx * 4 + 4] = val.w;
    }

    // 2. Load Halos (Apron)
    // We need to handle boundary conditions carefully using scalar loads for edges
    // Note: A fully optimized version would vectorize top/bottom loads too, 
    // but scalar loads for halos are safer for avoiding segfaults at image boundaries.

    // Load Top Halo
    if (ty == 0) {
        int y_load = max(0, iy - 1);
        for (int k = 0; k < 4; ++k) {
            int x_load = min(nx - 1, max(0, ix + k));
            s_mem[0][tx * 4 + 1 + k] = a[y_load * nx + x_load];
        }
    }
    // Load Bottom Halo
    if (ty == blockDim.y - 1) {
        int y_load = min(ny - 1, iy + 1);
        for (int k = 0; k < 4; ++k) {
            int x_load = min(nx - 1, max(0, ix + k));
            s_mem[BLOCK_H + 1][tx * 4 + 1 + k] = a[y_load * nx + x_load];
        }
    }
    // Load Left Halo (only thread 0 in x)
    if (tx == 0) {
        for (int k = 0; k < BLOCK_H + 2; ++k) { // Iterate vertically through shared mem logic
           // Map shared mem row k to global y
           int y_row = iy - 1 + k; 
           y_row = min(ny - 1, max(0, y_row));
           int x_load = max(0, ix - 1);
           s_mem[k][0] = a[y_row * nx + x_load];
        }
    }
    // Load Right Halo (only last thread in x)
    if (tx == blockDim.x - 1) {
        for (int k = 0; k < BLOCK_H + 2; ++k) {
           int y_row = iy - 1 + k;
           y_row = min(ny - 1, max(0, y_row));
           int x_load = min(nx - 1, ix + 4); // Pixel after the vector
           s_mem[k][BLOCK_W * 4 + 1] = a[y_row * nx + x_load];
        }
    }

    __syncthreads();

    // 3. Perform Convolution
    if (ix < nx && iy < ny) {
        uchar4 out_val;
        unsigned char* res = (unsigned char*)&out_val;

        // Process 4 pixels
        for(int k=0; k<4; ++k) {
            int s_x = tx * 4 + 1 + k;
            int s_y = ty + 1;

            float v = 
                c_filter[0] * s_mem[s_y - 1][s_x - 1] + c_filter[1] * s_mem[s_y - 1][s_x] + c_filter[2] * s_mem[s_y - 1][s_x + 1] +
                c_filter[3] * s_mem[s_y][s_x - 1]     + c_filter[4] * s_mem[s_y][s_x]     + c_filter[5] * s_mem[s_y][s_x + 1] +
                c_filter[6] * s_mem[s_y + 1][s_x - 1] + c_filter[7] * s_mem[s_y + 1][s_x] + c_filter[8] * s_mem[s_y + 1][s_x + 1];

            v = fminf(255.0f, fmaxf(0.0f, v + 0.5f)); // Clamp
            res[k] = (unsigned char)v;
        }

        // 4. Vectorized Store
        ((uchar4*)b)[idx_vec] = out_val;
    }
}

/**
 * @brief Implementation that uses shared memory for faster memory access and constant memory for the filter 
 * @param a input image
 * @param b output image
 * @param nx image width
 * @param nx image length
 */
__global__ void filter_shared(unsigned char *a, unsigned char *b, int nx, int ny) {
    // Shared memory with 1-pixel halo on all sides
    __shared__ unsigned char s_mem[BLOCK_H + 2][BLOCK_W + 2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int ix = blockIdx.x * blockDim.x + tx;
    int iy = blockIdx.y * blockDim.y + ty;

    // Load Center
    // We use clamping (min/max) to handle image edges (Neumann boundary condition)
    int load_x = min(nx - 1, max(0, ix));
    int load_y = min(ny - 1, max(0, iy));
    s_mem[ty + 1][tx + 1] = a[load_y * nx + load_x];

    // Load Left Halo
    if (tx == 0) {
        load_x = min(nx - 1, max(0, ix - 1));
        s_mem[ty + 1][0] = a[load_y * nx + load_x];
    }
    // Load Right Halo
    if (tx == blockDim.x - 1) {
        load_x = min(nx - 1, max(0, ix + 1));
        s_mem[ty + 1][BLOCK_W + 1] = a[load_y * nx + load_x];
    }
    // Load Top Halo
    if (ty == 0) {
        load_y = min(ny - 1, max(0, iy - 1));
        load_x = min(nx - 1, max(0, ix)); // Center x
        s_mem[0][tx + 1] = a[load_y * nx + load_x];
        
        // Corners for Top
        if(tx == 0) {
             int cx = min(nx - 1, max(0, ix - 1));
             s_mem[0][0] = a[load_y * nx + cx];
        }
        if(tx == blockDim.x - 1) {
             int cx = min(nx - 1, max(0, ix + 1));
             s_mem[0][BLOCK_W + 1] = a[load_y * nx + cx];
        }
    }
    // Load Bottom Halo
    if (ty == blockDim.y - 1) {
        load_y = min(ny - 1, max(0, iy + 1));
        load_x = min(nx - 1, max(0, ix));
        s_mem[BLOCK_H + 1][tx + 1] = a[load_y * nx + load_x];

        // Corners for Bottom
        if(tx == 0) {
            int cx = min(nx - 1, max(0, ix - 1));
            s_mem[BLOCK_H + 1][0] = a[load_y * nx + cx];
        }
        if(tx == blockDim.x - 1) {
            int cx = min(nx - 1, max(0, ix + 1));
            s_mem[BLOCK_H + 1][BLOCK_W + 1] = a[load_y * nx + cx];
        }
    }

    __syncthreads();

    if (ix < nx && iy < ny) {
        float v = 
            c_filter[0] * s_mem[ty][tx]         + c_filter[1] * s_mem[ty][tx + 1]     + c_filter[2] * s_mem[ty][tx + 2] +
            c_filter[3] * s_mem[ty + 1][tx]     + c_filter[4] * s_mem[ty + 1][tx + 1] + c_filter[5] * s_mem[ty + 1][tx + 2] +
            c_filter[6] * s_mem[ty + 2][tx]     + c_filter[7] * s_mem[ty + 2][tx + 1] + c_filter[8] * s_mem[ty + 2][tx + 2];

        v = fminf(255.0f, fmaxf(0.0f, v + 0.5f));
        b[iy * nx + ix] = (unsigned char)v;
    }
}

/**
 * @brief CPU implementation for the 2-D filter
 * @param a input data
 * @param b output data
 * @param c filter
 * @param nx image width
 * @param ny image length
 */
void filter_CPU(const std::vector<unsigned char> &a,
                std::vector<unsigned char> &b, int nx, int ny,
                const std::vector<float> &c) {
  auto idx = [&nx](int y, int x) { return y * nx + x; };

  for (int y = 0; y < ny; ++y) {
    for (int x = 0; x < nx; ++x) {
      int xl = std::max(0, x - 1);
      int yl = std::max(0, y - 1);
      int xh = std::min(nx - 1, x + 1);
      int yh = std::min(ny - 1, y + 1);

      float v =
          c[0] * a[idx(yl, xl)] + c[1] * a[idx(yl, x)] + c[2] * a[idx(yl, xh)] +
          c[3] * a[idx(y, xl)] + c[4] * a[idx(y, x)] + c[5] * a[idx(y, xh)] +
          c[6] * a[idx(yh, xl)] + c[7] * a[idx(yh, x)] + c[8] * a[idx(yh, xh)];

      uint f = (uint)(v + 0.5f);
      b[idx(y, x)] =
          (unsigned char)std::min(255, std::max(0, static_cast<int>(f)));
    }
  }
}

int main() {
  CALI_CXX_MARK_FUNCTION;

  // Create caliper ConfigManager object 
  cali::ConfigManager mgr;
  mgr.start();

  // Image size
  int nx = 4096; // Adjusted to test larger sizes as suggested in PDF, or keep 1024
  int ny = 4096;
  const int size = nx * ny;
  const int size_bytes = size * sizeof(unsigned char);

  // Allocate host memory
  std::vector<unsigned char> input_img(size);
  std::vector<unsigned char> output_img_shared(size);
  std::vector<unsigned char> output_img_vector(size);
  std::vector<unsigned char> output_img_ref(size);

  // Initialize input image with some test data
  for (int i = 0; i < size; ++i) {
    input_img[i] = static_cast<unsigned char>(i % 256);
  }

  // Define Filter (e.g., Blur or Identity)
  // Simple Box Blur 3x3
  std::vector<float> host_filter = {
      1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f,
      1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f,
      1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f
  };

  // Allocate device memory
  unsigned char *d_in, *d_out_shared, *d_out_vector;
  cudaMalloc((void**)&d_in, size_bytes);
  cudaMalloc((void**)&d_out_shared, size_bytes);
  cudaMalloc((void**)&d_out_vector, size_bytes);

  // Copy input data to device
  CALI_MARK_BEGIN("cudaMemcpy_HostToDevice");
  cudaMemcpy(d_in, input_img.data(), size_bytes, cudaMemcpyHostToDevice);
  CALI_MARK_END("cudaMemcpy_HostToDevice");

  // Copy filter coefficients to constant memory
  cudaMemcpyToSymbol(c_filter, host_filter.data(), 9 * sizeof(float));

  // --- Run Shared Memory Kernel ---
  dim3 dimBlock(BLOCK_W, BLOCK_H);
  dim3 dimGrid((nx + dimBlock.x - 1) / dimBlock.x, (ny + dimBlock.y - 1) / dimBlock.y);

  CALI_MARK_BEGIN("filter_shared");
  filter_shared<<<dimGrid, dimBlock>>>(d_in, d_out_shared, nx, ny);
  cudaDeviceSynchronize();
  CALI_MARK_END("filter_shared");

  // Copy result back to host
  CALI_MARK_BEGIN("cudaMemcpy_DeviceToHost");
  cudaMemcpy(output_img_shared.data(), d_out_shared, size_bytes, cudaMemcpyDeviceToHost);
  CALI_MARK_END("cudaMemcpy_DeviceToHost");


  // --- Run Vectorized Shared Memory Kernel ---
  // Note: Grid width is divided by 4 because each thread processes 4 elements
  dim3 dimGridVec((nx/4 + dimBlock.x - 1) / dimBlock.x, (ny + dimBlock.y - 1) / dimBlock.y);

  CALI_MARK_BEGIN("filter_shared_vectorized");
  filter_shared_vector<<<dimGridVec, dimBlock>>>(d_in, d_out_vector, nx, ny);
  cudaDeviceSynchronize();
  CALI_MARK_END("filter_shared_vectorized");

  // Copy result back to host (Vectorized output)
  cudaMemcpy(output_img_vector.data(), d_out_vector, size_bytes, cudaMemcpyDeviceToHost);

  // Run CPU version for comparison
  std::cout << "Running CPU reference..." << std::endl;
  filter_CPU(input_img, output_img_ref, nx, ny, host_filter);

  // Check result by comparing cpu and gpu
  bool correct_shared = true;
  bool correct_vector = true;

  for(int i = 0; i < size; i++) {
      if(output_img_ref[i] != output_img_shared[i]) {
          // Allow small differences due to float precision
          if (abs((int)output_img_ref[i] - (int)output_img_shared[i]) > 1) {
              correct_shared = false;
              // std::cout << "Mismatch Shared at " << i << ": CPU " << (int)output_img_ref[i] << " GPU " << (int)output_img_shared[i] << std::endl;
              // break;
          }
      }
      if(output_img_ref[i] != output_img_vector[i]) {
          if (abs((int)output_img_ref[i] - (int)output_img_vector[i]) > 1) {
             correct_vector = false;
             // std::cout << "Mismatch Vector at " << i << ": CPU " << (int)output_img_ref[i] << " GPU " << (int)output_img_vector[i] << std::endl;
             // break;
          }
      }
  }

  if (correct_shared) std::cout << "Shared Memory Kernel: PASS" << std::endl;
  else std::cout << "Shared Memory Kernel: FAIL" << std::endl;

  if (correct_vector) std::cout << "Vectorized Kernel: PASS" << std::endl;
  else std::cout << "Vectorized Kernel: FAIL" << std::endl;

  // Cleanup
  cudaFree(d_in);
  cudaFree(d_out_shared);
  cudaFree(d_out_vector);

  // Flush Caliper output
  mgr.stop();
  mgr.flush();

  std::cout << "End" << "\n";
  return 0;
}