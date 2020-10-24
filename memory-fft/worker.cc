#include <hbwmalloc.h>
#include <mkl.h>
#include <omp.h>
#include <stdio.h>
#include <string.h>

// implement scratch buffer on HBM and compute FFTs, refer instructions on Lab
// page
void runFFTs(const size_t fft_size, const size_t num_fft, MKL_Complex8 *data,
             DFTI_DESCRIPTOR_HANDLE *fftHandle) {
  // printf("Llega\n");
  // printf("%f\n", data[12]);
  MKL_Complex8 *buff;
  hbw_posix_memalign((void **)&buff, 4096, sizeof(MKL_Complex8) * fft_size);
  // printf("%f\n", buff[12]);
  for (size_t i = 0; i < num_fft; i++) {
#pragma omp parallel for
    for (size_t j = 0; j < fft_size; j++) {
      buff[j].real = data[j + i * fft_size].real;
      buff[j].imag = data[j + i * fft_size].imag;
    }
    DftiComputeForward(*fftHandle, &buff[0]);
#pragma omp parallel for
    for (size_t j = 0; j < fft_size; j++) {
      data[j + i * fft_size].real = buff[j].real;
      data[j + i * fft_size].imag = buff[j].imag;
    }
  }
  // printf("%f\n", buff[12]);
  hbw_free(buff);
  // printf("%f\n", data[12]);
}
