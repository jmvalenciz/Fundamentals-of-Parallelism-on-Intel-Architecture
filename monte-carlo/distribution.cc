#include "distribution.h"

//distribution function definition
float dist_func(const float alpha, const float rn) {
  return delta_max*sinf(alpha*rn)*expf(-rn*rn);
}