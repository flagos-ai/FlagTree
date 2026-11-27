
__attribute__((device)) void fnx(int *state) {
  if (threadIdx.x == 0) {
    state[1] = state[0];
    state[0] += 1 << 1;
  }
}

__attribute__((device)) void fn(int *state) {
  if (threadIdx.x == 0) {
    state[1] = state[0];
    state[0] += 1 << 3;
  }
}

namespace symbols {

__attribute__((device)) void fn(int *state) {
  if (threadIdx.x == 0) {
    state[1] = state[0];
    state[0] += 1 << 7;
  }
}

__attribute__((device)) void fnx(int *state) {
  if (threadIdx.x == 0) {
    state[1] = state[0];
    state[0] += 1 << 5;
  }
}

} // namespace symbols

// Keep a real helper call after the generated wrapper inlines accumulate.
static __attribute__((device, noinline)) void update(int *state) {
  state[2] = state[1];
  state[1] = state[0];
  state[0] += 3;
}

__attribute__((device)) void accumulate(int *state) {
  if (threadIdx.x == 0)
    update(state);
}
