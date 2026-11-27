
namespace symbols {

__attribute__((device)) void fnx(int *state) {
  if (threadIdx.x == 0) {
    state[1] = state[0];
    state[0] += 1 << 4;
  }
}

__attribute__((device)) void fn(int *state) {
  if (threadIdx.x == 0) {
    state[1] = state[0];
    state[0] += 1 << 6;
  }
}

} // namespace symbols

__attribute__((device)) void fnx(int *state) {
  if (threadIdx.x == 0) {
    state[1] = state[0];
    state[0] += 1 << 0;
  }
}

__attribute__((device)) void fn(int *state) {
  if (threadIdx.x == 0) {
    state[1] = state[0];
    state[0] += 1 << 2;
  }
}

// Keep a real helper call after the generated wrapper inlines accumulate.
static __attribute__((device, noinline)) void update(int *state) {
  state[2] = state[1];
  state[1] = state[0];
  state[0] += 1;
}

__attribute__((device)) void accumulate(int *state) {
  if (threadIdx.x == 0)
    update(state);
}
