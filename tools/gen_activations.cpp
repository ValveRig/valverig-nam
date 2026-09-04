// Regenerates assets/activations.f32 from the C++ reference's own headers.
//
// 8001 records of 8 floats, at x = i * 0.002 for i in [-4000, 4000]:
//   fast_tanh, sigmoid, swish, hardswish, softsign, tanh, hard_tanh,
//   leaky_hardtanh(-0.5, 0.75, 0.02, 0.03)
//
// The order and the input grid are pinned by tests/activations.rs.
#include <cmath>
#include <cstdio>
#include "NAM/activations.h"

int main(int argc, char** argv)
{
  if (argc < 2) { std::fprintf(stderr, "usage: gen_activations <out.f32>\n"); return 2; }
  FILE* f = std::fopen(argv[1], "wb");
  if (!f) { std::perror(argv[1]); return 1; }
  for (int i = -4000; i <= 4000; i++)
  {
    const float x = (float)i * 0.002f;
    const float v[8] = {
      nam::activations::fast_tanh(x),
      nam::activations::sigmoid(x),
      nam::activations::swish(x),
      nam::activations::hardswish(x),
      nam::activations::softsign(x),
      std::tanh(x),
      nam::activations::hard_tanh(x),
      nam::activations::leaky_hardtanh(x, -0.5f, 0.75f, 0.02f, 0.03f),
    };
    std::fwrite(v, sizeof(float), 8, f);
  }
  std::fclose(f);
  return 0;
}
