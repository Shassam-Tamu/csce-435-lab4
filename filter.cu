#include <iostream>
#include <stdio.h>
#include <vector>

#include <adiak.hpp>
#include <caliper/cali-manager.h>
#include <caliper/cali.h>

__constant__ float c_filter[25];

#define BLOCK_WIDTH 16
#define BLOCK_HEIGHT 16
#define FILTER_RADIUS 2


/**
 * @brief Implementation that uses shared memory for faster memory access along with vectorized operations 
 *  and constant memory for the filter 
 * @param a input image
 * @param b output image
 * @param nx image width
 * @param nx image length
 */
__global__ void filter_shared_vector(unsigned char *a, unsigned char *b, int nx,int ny) {
  __shared__ unsigned char shared_mem[BLOCK_HEIGHT + 4][BLOCK_WIDTH * 4 + 4];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int ix = (blockIdx.x * blockDim.x + tx) * 4;
  int iy = blockIdx.y * blockDim.y + ty;

  // 1D index for vectorized load/store
  int idx_vec = iy * (nx / 4) + (blockIdx.x * blockDim.x + tx);

  // Load Main Data Using Vectorized Load
  if (ix < nx && iy < ny) {
    uchar4 val = ((uchar4*)a)[idx_vec];
    // Unpack into shared memory, offset by 2 for halo
    shared_mem[ty + FILTER_RADIUS][tx * 4 + FILTER_RADIUS] = val.x;
    shared_mem[ty + FILTER_RADIUS][tx * 4 + FILTER_RADIUS + 1] = val.y;
    shared_mem[ty + FILTER_RADIUS][tx * 4 + FILTER_RADIUS + 2] = val.z;
    shared_mem[ty + FILTER_RADIUS][tx * 4 + FILTER_RADIUS + 3] = val.w;
  }

  // Load Halo Regions - need 2 rows/cols on each side for 5x5 filter
  // Top Halo (2 rows)
  if (ty < FILTER_RADIUS) {
    for (int row = 0; row < FILTER_RADIUS; ++row) {
      if (ty == row) {
        int y_load = max(0, iy - FILTER_RADIUS + row);
        for (int k = 0; k < 4; ++k) {
          int x_load = min(nx - 1, max(0, ix + k));
          shared_mem[row][tx * 4 + FILTER_RADIUS + k] = a[y_load * nx + x_load];
        }
      }
    }
  }
  // Bottom Halo (2 rows)
  if (ty >= blockDim.y - FILTER_RADIUS) {
    for (int row = 0; row < FILTER_RADIUS; ++row) {
      if (ty == blockDim.y - FILTER_RADIUS + row) {
        int y_load = min(ny - 1, iy + FILTER_RADIUS - row);
        for (int k = 0; k < 4; ++k) {
          int x_load = min(nx - 1, max(0, ix + k));
          shared_mem[BLOCK_HEIGHT + FILTER_RADIUS + row][tx * 4 + FILTER_RADIUS + k] = a[y_load * nx + x_load];
        }
      }
    }
  }
  // Left Halo (2 cols)
  if (tx == 0) {
    for (int k = 0; k < BLOCK_HEIGHT + 2 * FILTER_RADIUS; ++k) {
      int y_row = iy - FILTER_RADIUS + k;
      y_row = min(ny - 1, max(0, y_row));
      for (int col = 0; col < FILTER_RADIUS; ++col) {
        int x_load = max(0, ix - FILTER_RADIUS + col);
        shared_mem[k][col] = a[y_row * nx + x_load];
      }
    }
  }
  // Right Halo (2 cols)
  if (tx == blockDim.x - 1) {
    for (int k = 0; k < BLOCK_HEIGHT + 2 * FILTER_RADIUS; ++k) {
      int y_row = iy - FILTER_RADIUS + k;
      y_row = min(ny - 1, max(0, y_row));
      for (int col = 0; col < FILTER_RADIUS; ++col) {
        int x_load = min(nx - 1, ix + 4 + col);
        shared_mem[k][BLOCK_WIDTH * 4 + FILTER_RADIUS + col] = a[y_row * nx + x_load];
      }
    }
  }

  __syncthreads();

  // Apply 5x5 Filter
  if (ix < nx && iy < ny) {
    uchar4 output_val;
    unsigned char* result = (unsigned char*)&output_val;

    for (int k = 0; k < 4; ++k) {
      int s_x = tx * 4 + FILTER_RADIUS + k;
      int s_y = ty + FILTER_RADIUS;
      float v = 0.0f;
      for (int fy = -FILTER_RADIUS; fy <= FILTER_RADIUS; ++fy) {
        for (int fx = -FILTER_RADIUS; fx <= FILTER_RADIUS; ++fx) {
          int filter_idx = (fy + FILTER_RADIUS) * 5 + (fx + FILTER_RADIUS);
          v += c_filter[filter_idx] * shared_mem[s_y + fy][s_x + fx];
        }
      }
      v = min(max(v, 0.0f), 255.0f);
      result[k] = (unsigned char)v;
    }

    ((uchar4*)b)[idx_vec] = output_val;
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
  __shared__ unsigned char shared_mem[BLOCK_HEIGHT + 4][BLOCK_WIDTH + 4];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int ix = blockIdx.x * blockDim.x + tx;
  int iy = blockIdx.y * blockDim.y + ty;

  // Load Center
  int load_x = min(nx - 1, max(0, ix));
  int load_y = min(ny - 1, max(0, iy));
  shared_mem[ty + FILTER_RADIUS][tx + FILTER_RADIUS] = a[load_y * nx + load_x];

  // Load Halo Regions - need 2 pixels on each side for 5x5 filter
  // Load Left Halo (2 columns)
  if (tx < FILTER_RADIUS) {
    for (int col = 0; col < FILTER_RADIUS; ++col) {
      if (tx == col) {
        load_x = min(nx - 1, max(0, ix - FILTER_RADIUS + col));
        shared_mem[ty + FILTER_RADIUS][col] = a[load_y * nx + load_x];
      }
    }
  }
  // Load Right Halo (2 columns)
  if (tx >= blockDim.x - FILTER_RADIUS) {
    for (int col = 0; col < FILTER_RADIUS; ++col) {
      if (tx == blockDim.x - FILTER_RADIUS + col) {
        load_x = min(nx - 1, ix + FILTER_RADIUS - col);
        shared_mem[ty + FILTER_RADIUS][BLOCK_WIDTH + FILTER_RADIUS + col] = a[load_y * nx + load_x];
      }
    }
  }
  // Load Top Halo (2 rows)
  if (ty < FILTER_RADIUS) {
    for (int row = 0; row < FILTER_RADIUS; ++row) {
      if (ty == row) {
        load_y = min(ny - 1, max(0, iy - FILTER_RADIUS + row));
        load_x = min(nx - 1, max(0, ix));
        shared_mem[row][tx + FILTER_RADIUS] = a[load_y * nx + load_x];
        // Load Top-Left Corner
        if (tx < FILTER_RADIUS) {
          for (int col = 0; col < FILTER_RADIUS; ++col) {
            if (tx == col) {
              int cx = min(nx - 1, max(0, ix - FILTER_RADIUS + col));
              shared_mem[row][col] = a[load_y * nx + cx];
            }
          }
        }
        // Load Top-Right Corner
        if (tx >= blockDim.x - FILTER_RADIUS) {
          for (int col = 0; col < FILTER_RADIUS; ++col) {
            if (tx == blockDim.x - FILTER_RADIUS + col) {
              int cx = min(nx - 1, ix + FILTER_RADIUS - col);
              shared_mem[row][BLOCK_WIDTH + FILTER_RADIUS + col] = a[load_y * nx + cx];
            }
          }
        }
      }
    }
  }
  // Load Bottom Halo (2 rows)
  if (ty >= blockDim.y - FILTER_RADIUS) {
    for (int row = 0; row < FILTER_RADIUS; ++row) {
      if (ty == blockDim.y - FILTER_RADIUS + row) {
        load_y = min(ny - 1, iy + FILTER_RADIUS - row);
        load_x = min(nx - 1, max(0, ix));
        shared_mem[BLOCK_HEIGHT + FILTER_RADIUS + row][tx + FILTER_RADIUS] = a[load_y * nx + load_x];
        // Load Bottom-Left Corner
        if (tx < FILTER_RADIUS) {
          for (int col = 0; col < FILTER_RADIUS; ++col) {
            if (tx == col) {
              int cx = min(nx - 1, max(0, ix - FILTER_RADIUS + col));
              shared_mem[BLOCK_HEIGHT + FILTER_RADIUS + row][col] = a[load_y * nx + cx];
            }
          }
        }
        // Load Bottom-Right Corner
        if (tx >= blockDim.x - FILTER_RADIUS) {
          for (int col = 0; col < FILTER_RADIUS; ++col) {
            if (tx == blockDim.x - FILTER_RADIUS + col) {
              int cx = min(nx - 1, ix + FILTER_RADIUS - col);
              shared_mem[BLOCK_HEIGHT + FILTER_RADIUS + row][BLOCK_WIDTH + FILTER_RADIUS + col] = a[load_y * nx + cx];
            }
          }
        }
      }
    }
  }
  __syncthreads();

  // Apply 5x5 Filter
  if (ix < nx && iy < ny) {
    float v = 0.0f;
    for (int fy = -FILTER_RADIUS; fy <= FILTER_RADIUS; ++fy) {
      for (int fx = -FILTER_RADIUS; fx <= FILTER_RADIUS; ++fx) {
        int filter_idx = (fy + FILTER_RADIUS) * 5 + (fx + FILTER_RADIUS);
        v += c_filter[filter_idx] * shared_mem[ty + FILTER_RADIUS + fy][tx + FILTER_RADIUS + fx];
      }
    }
    v = min(max(v, 0.0f), 255.0f);
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
  const int radius = 2;  // 5x5 filter has radius 2

  for (int y = 0; y < ny; ++y) {
    for (int x = 0; x < nx; ++x) {
      float v = 0.0f;
      for (int fy = -radius; fy <= radius; ++fy) {
        for (int fx = -radius; fx <= radius; ++fx) {
          int sample_y = std::min(ny - 1, std::max(0, y + fy));
          int sample_x = std::min(nx - 1, std::max(0, x + fx));
          int filter_idx = (fy + radius) * 5 + (fx + radius);
          v += c[filter_idx] * a[idx(sample_y, sample_x)];
        }
      }
      uint f = (uint)(v + 0.5f);
      b[idx(y, x)] =
          (unsigned char)std::min(255, std::max(0, static_cast<int>(f)));
    }
  }
}

int main(int argc, char* argv[]) {
  CALI_CXX_MARK_FUNCTION;

  // Create caliper ConfigManager object 
  cali::ConfigManager mgr;
  mgr.start();

  // Image size from command line (default 1024)
  int img_size = 1024;
  if (argc > 1) {
    img_size = atoi(argv[1]);
  }
  int nx = img_size;
  int ny = img_size;
  const int size = nx * ny;
  const size_t size_bytes = size * sizeof(unsigned char);

  // Filter size (5x5)
  const int filter_size = 5;

  // Initialize adiak
  adiak::init(NULL);
  adiak::value("image_width", nx);
  adiak::value("image_height", ny);
  adiak::value("image_size_total", size);
  adiak::value("filter_size", filter_size);
  adiak::value("block_width", BLOCK_WIDTH);
  adiak::value("block_height", BLOCK_HEIGHT);

  std::cout << "Image size: " << nx << "x" << ny << " (" << size << " pixels)" << std::endl;
  std::cout << "Filter size: " << filter_size << "x" << filter_size << std::endl;

  // Allocate host memory
  std::vector<unsigned char> input_img(size);
  std::vector<unsigned char> output_img_shared(size);
  std::vector<unsigned char> output_img_vector(size);
  std::vector<unsigned char> output_img_ref(size);
  // Initialize input image with some test data
  for (int i = 0; i < size; ++i) {
    input_img[i] = static_cast<unsigned char>(i % 256);
  }
  std::vector<float> host_filter = {
      1.0f/25.0f, 1.0f/25.0f, 1.0f/25.0f, 1.0f/25.0f, 1.0f/25.0f,
      1.0f/25.0f, 1.0f/25.0f, 1.0f/25.0f, 1.0f/25.0f, 1.0f/25.0f,
      1.0f/25.0f, 1.0f/25.0f, 1.0f/25.0f, 1.0f/25.0f, 1.0f/25.0f,
      1.0f/25.0f, 1.0f/25.0f, 1.0f/25.0f, 1.0f/25.0f, 1.0f/25.0f,
      1.0f/25.0f, 1.0f/25.0f, 1.0f/25.0f, 1.0f/25.0f, 1.0f/25.0f
  };

  // Create CUDA events for timing
  cudaEvent_t start_h2d, stop_h2d, start_d2h, stop_d2h;
  cudaEvent_t start_shared, stop_shared, start_vector, stop_vector;
  cudaEventCreate(&start_h2d);
  cudaEventCreate(&stop_h2d);
  cudaEventCreate(&start_d2h);
  cudaEventCreate(&stop_d2h);
  cudaEventCreate(&start_shared);
  cudaEventCreate(&stop_shared);
  cudaEventCreate(&start_vector);
  cudaEventCreate(&stop_vector);

  // Allocate device memory
  unsigned char *d_in, *d_out_shared, *d_out_vector;
  cudaMalloc((void**)&d_in, size_bytes);
  cudaMalloc((void**)&d_out_shared, size_bytes);
  cudaMalloc((void**)&d_out_vector, size_bytes);

  // Copy input data to device with timing
  CALI_MARK_BEGIN("cudaMemcpy_HostToDevice");
  cudaEventRecord(start_h2d);
  cudaMemcpy(d_in, input_img.data(), size_bytes, cudaMemcpyHostToDevice);
  cudaEventRecord(stop_h2d);
  cudaEventSynchronize(stop_h2d);
  CALI_MARK_END("cudaMemcpy_HostToDevice");

  float time_h2d_ms = 0;
  cudaEventElapsedTime(&time_h2d_ms, start_h2d, stop_h2d);
  float bandwidth_h2d = (size_bytes / (1e9)) / (time_h2d_ms / 1000.0f); // GB/s

  std::cout << "Host to Device transfer time: " << time_h2d_ms << " ms" << std::endl;
  std::cout << "Host to Device effective bandwidth: " << bandwidth_h2d << " GB/s" << std::endl;

  // Copy filter coefficients to constant memory
  cudaMemcpyToSymbol(c_filter, host_filter.data(), 25 * sizeof(float));

  // Launch shared memory kernel
  dim3 dimBlock(BLOCK_WIDTH, BLOCK_HEIGHT);
  dim3 dimGrid((nx + dimBlock.x - 1) / dimBlock.x, (ny + dimBlock.y - 1) / dimBlock.y);

  CALI_MARK_BEGIN("filter_shared");
  cudaEventRecord(start_shared);
  filter_shared<<<dimGrid, dimBlock>>>(d_in, d_out_shared, nx, ny);
  cudaEventRecord(stop_shared);
  cudaEventSynchronize(stop_shared);
  CALI_MARK_END("filter_shared");

  float time_shared_ms = 0;
  cudaEventElapsedTime(&time_shared_ms, start_shared, stop_shared);
  std::cout << "Shared memory kernel time: " << time_shared_ms << " ms" << std::endl;

  // Copy result back to host with timing
  CALI_MARK_BEGIN("cudaMemcpy_DeviceToHost");
  cudaEventRecord(start_d2h);
  cudaMemcpy(output_img_shared.data(), d_out_shared, size_bytes, cudaMemcpyDeviceToHost);
  cudaEventRecord(stop_d2h);
  cudaEventSynchronize(stop_d2h);
  CALI_MARK_END("cudaMemcpy_DeviceToHost");

  float time_d2h_ms = 0;
  cudaEventElapsedTime(&time_d2h_ms, start_d2h, stop_d2h);
  float bandwidth_d2h = (size_bytes / (1e9)) / (time_d2h_ms / 1000.0f); // GB/s

  std::cout << "Device to Host transfer time: " << time_d2h_ms << " ms" << std::endl;
  std::cout << "Device to Host effective bandwidth: " << bandwidth_d2h << " GB/s" << std::endl;

  // Launch shared memory vectorized kernel
  dim3 dimGridVec((nx/4 + dimBlock.x - 1) / dimBlock.x, (ny + dimBlock.y - 1) / dimBlock.y);

  CALI_MARK_BEGIN("filter_shared_vectorized");
  cudaEventRecord(start_vector);
  filter_shared_vector<<<dimGridVec, dimBlock>>>(d_in, d_out_vector, nx, ny);
  cudaEventRecord(stop_vector);
  cudaEventSynchronize(stop_vector);
  CALI_MARK_END("filter_shared_vectorized");

  float time_vector_ms = 0;
  cudaEventElapsedTime(&time_vector_ms, start_vector, stop_vector);
  std::cout << "Vectorized kernel time: " << time_vector_ms << " ms" << std::endl;

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
          }
      }
      if(output_img_ref[i] != output_img_vector[i]) {
          if (abs((int)output_img_ref[i] - (int)output_img_vector[i]) > 1) {
             correct_vector = false;
          }
      }
  }

  if (correct_shared) std::cout << "Shared Memory Kernel: PASS" << std::endl;
  else std::cout << "Shared Memory Kernel: FAIL" << std::endl;

  if (correct_vector) std::cout << "Vectorized Kernel: PASS" << std::endl;
  else std::cout << "Vectorized Kernel: FAIL" << std::endl;

  // Record timing values with adiak
  adiak::value("time_h2d_ms", time_h2d_ms);
  adiak::value("time_d2h_ms", time_d2h_ms);
  adiak::value("time_shared_kernel_ms", time_shared_ms);
  adiak::value("time_vectorized_kernel_ms", time_vector_ms);
  adiak::value("bandwidth_h2d_GBps", bandwidth_h2d);
  adiak::value("bandwidth_d2h_GBps", bandwidth_d2h);
  adiak::value("data_size_bytes", (long long)size_bytes);

  // Cleanup CUDA events
  cudaEventDestroy(start_h2d);
  cudaEventDestroy(stop_h2d);
  cudaEventDestroy(start_d2h);
  cudaEventDestroy(stop_d2h);
  cudaEventDestroy(start_shared);
  cudaEventDestroy(stop_shared);
  cudaEventDestroy(start_vector);
  cudaEventDestroy(stop_vector);

  // Cleanup device memory
  cudaFree(d_in);
  cudaFree(d_out_shared);
  cudaFree(d_out_vector);

  // Flush Caliper output
  mgr.stop();
  mgr.flush();

  // Finalize adiak
  adiak::fini();

  std::cout << "End" << "\n";
  return 0;
}
