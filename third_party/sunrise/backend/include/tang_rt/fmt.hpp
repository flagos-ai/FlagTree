#ifndef _TANGRT_FMT_HPP_
#define _TANGRT_FMT_HPP_

#include <stdint.h>

#include <limits>
#include <type_traits>
#include <utility>

// #define PT_PRINTF_ENDMARKER
#define PT_PRINTF_READY

#if !defined(__TANGC_MAJOR__) && !defined(__device__)
#define __device__
#endif  //!< __TANGC_MAJOR__

#ifndef unlikely
#define unlikely(x) __builtin_expected(!!(x), 0)
#endif  //!< unlikely

#ifndef lower_32_bit
#define lower_32_bit(x) (((uint64_t)x) & 0xffffffff)
#endif

#ifndef upper_32_bit
#define upper_32_bit(x) (((uint64_t)x) >> 32)
#endif

namespace tangrt {
namespace fmt {

static constexpr unsigned kArgAlignment     = 4;
static constexpr unsigned kArgAlignmentMask = ~(kArgAlignment - 1);

enum ArgId {
  ArgId_None = 0,

  ArgId_char  = 1,
  ArgId_schar = 2,
  ArgId_uchar = 3,

  ArgId_short  = 5,
  ArgId_ushort = 6,

  ArgId_int  = 7,
  ArgId_uint = 8,

  ArgId_long  = 9,
  ArgId_ulong = 10,

  ArgId_longlong  = 11,
  ArgId_ulonglong = 12,

  ArgId_float       = 13,
  ArgId_double      = 14,
  ArgId_long_double = 15,

  //!< nullptr
  ArgId_nullptr = 20,

  //!< generic pointer type.
  ArgId_pointer = 21,

  //!< char* ptr = nullptr;
  ArgId_char_nullptr  = 22,
  ArgId_schar_nullptr = 23,
  ArgId_uchar_nullptr = 24,

  ArgId_char_pointer  = 25,
  ArgId_schar_pointer = 26,
  ArgId_uchar_pointer = 27,

  //!< char[]
  ArgId_char_array  = 28,
  ArgId_schar_array = 29,
  ArgId_uchar_array = 30,
};

static inline const char* GetArgIdName(const int id) {
#define _case(x, str) \
  case x:             \
    return str

  switch (id) {
    _case(ArgId_char, "char");
    _case(ArgId_schar, "signed char");
    _case(ArgId_uchar, "unsigned char");

    _case(ArgId_short, "short");
    _case(ArgId_ushort, "unsigned short");
    _case(ArgId_int, "int");
    _case(ArgId_uint, "unsigned int");
    _case(ArgId_long, "long");
    _case(ArgId_ulong, "unsigned long");
    _case(ArgId_longlong, "long long");
    _case(ArgId_ulonglong, "unsigned long long");

    _case(ArgId_float, "float");
    _case(ArgId_double, "double");
    _case(ArgId_long_double, "long double");

    _case(ArgId_nullptr, "nullptr");
    _case(ArgId_pointer, "pointer");

    _case(ArgId_char_nullptr, "char nullptr");
    _case(ArgId_schar_nullptr, "schar nullptr");
    _case(ArgId_uchar_nullptr, "uchar nullptr");

    _case(ArgId_char_pointer, "char pointer");
    _case(ArgId_schar_pointer, "schar pointer");
    _case(ArgId_uchar_pointer, "uchar pointer");

    _case(ArgId_char_array, "char array");
    _case(ArgId_schar_array, "schar array");
    _case(ArgId_uchar_array, "uchar array");
    default:
      return "None";
  }
#undef _case
}

static __device__ inline unsigned ArgAlign(unsigned int x) {
  return (x + kArgAlignment - 1) & ~(kArgAlignment - 1);
}

static __device__ inline unsigned int ArgTraitsStrLen(const char* s) {
  const char* p = s;
  while (*p) {
    ++p;
  }
  return p - s;
}

namespace detail {

union u32c4_u {
  char     c[4];
  uint32_t u32;

  __device__ u32c4_u(char ch0, char ch1 = 0, char ch2 = 0, char ch3 = 0)
    : c{ch0, ch1, ch2, ch3} {}
};

template <typename T, unsigned N>
__device__ void FundamentalFillImpl(const uint32_t id,
                                    const T        t,
                                    uint32_t*      buf,
                                    uint32_t&      pos,
                                    const uint32_t mask) {
  static_assert(N == 1 || N == 2 || N == 4, "");
  static_assert(!std::is_same<char*, T>::value,
                "This helper function is not suitable for char*.");

  union {
    uint32_t u[N];
    T        t;
  } x;
  x.t               = t;
  buf[pos++ & mask] = (sizeof(T) << 16) | id;
  for (unsigned int i = 0; i < N; ++i) {
    buf[pos++ & mask] = x.u[i];
  }
}

template <typename T>
__device__ void FundamentalFill(const uint32_t id,
                                const T        t,
                                uint32_t*      buf,
                                uint32_t&      pos,
                                const uint32_t mask) {
  FundamentalFillImpl<T, sizeof(T) / sizeof(uint32_t)>(id, t, buf, pos, mask);
}

#if 0
template <>
__device__ void FundamentalFill(const uint32_t id, const float t, uint32_t* buf,
                     uint32_t& pos, const uint32_t mask)
{
  buf[pos++ & mask] = (sizeof(float) << 16) | id;
  union
  {
    uint32_t u[2];
    float f;
  } x;
  x.f = t;
  buf[pos++ & mask] = x.u[0];
}

//!< to avoid strict aliasing
//!< -fno-strict-aliasing
template <>
__device__ void FundamentalFill(const uint32_t id, const double t, uint32_t* buf,
                     uint32_t& pos, const uint32_t mask)
{
  buf[pos++ & mask] = (sizeof(double) << 16) | id;
  union
  {
    uint32_t u[2];
    double d;
  } x;
  x.d = t;
  buf[pos++ & mask] = x.u[0];
  buf[pos++ & mask] = x.u[1];
}
#endif

static __device__ inline void StringFillData(const char*     s,
                                             const uint32_t  sizeBytes,
                                             uint32_t* const buf,
                                             uint32_t&       pos,
                                             const uint32_t  mask) {
  const uint32_t* s32 = (const uint32_t*)s;
  for (unsigned int i = 0; i < sizeBytes / sizeof(uint32_t); ++i) {
    buf[pos++ & mask] = s32[i];
  }
  switch (sizeBytes & (4 - 1)) {
    case 1: {
      detail::u32c4_u x(s[sizeBytes - 1]);
      buf[pos++ & mask] = x.u32;
      break;
    };
    case 2: {
      detail::u32c4_u x(s[sizeBytes - 2], s[sizeBytes - 1]);
      buf[pos++ & mask] = x.u32;
      break;
    };
    case 3: {
      detail::u32c4_u x(s[sizeBytes - 3], s[sizeBytes - 2], s[sizeBytes - 1]);
      buf[pos++ & mask] = x.u32;
      break;
    }
  }
}  // StringFillData function

}  // namespace detail

template <typename T>
struct FmtTraits;

template <unsigned N>
struct FmtTraits<char[N]> {
  static __device__ unsigned int FmtLength(const char*) {
    return 4 + ArgAlign(N);
  }

  static __device__ void Fill(const char* s,
                              uint32_t*   buf,
                              uint32_t&   pos,
                              uint32_t    mask) {
    //!< Fill header
    buf[pos++ & mask] = (N << 16) | ArgId_schar_array;
    detail::StringFillData(s, N, buf, pos, mask);
  }
};

template <unsigned N>
struct FmtTraits<const char[N]> {
  static __device__ unsigned int FmtLength(const char*) {
    return 4 + ArgAlign(N);
  }

  static __device__ void Fill(const char* s,
                              uint32_t*   buf,
                              uint32_t&   pos,
                              uint32_t    mask) {
    //!< Fill header
    buf[pos++ & mask] = (N << 16) | ArgId_schar_array;
    detail::StringFillData(s, N, buf, pos, mask);
  }
};

template <>
struct FmtTraits<const char*> {
  static __device__ unsigned int FmtLength(const char* s) {
    return 4 + (s ? ArgAlign(ArgTraitsStrLen(s)) : 0);
  }

  static __device__ void Fill(const char* s,
                              uint32_t*   buf,
                              uint32_t&   pos,
                              uint32_t    mask) {
    if (!s) {
      buf[pos++ & mask] = ArgId_char_nullptr;
      return;
    }
    const uint32_t N = ArgTraitsStrLen(s);
    detail::StringFillData(s, N, buf, pos, mask);
  }
};

template <>
struct FmtTraits<char*> {
  static __device__ unsigned int FmtLength(const char* s) {
    return 4 + (s ? ArgAlign(ArgTraitsStrLen(s)) : 0);
  }

  static __device__ void Fill(const char* s,
                              uint32_t*   buf,
                              uint32_t    pos,
                              uint32_t    mask) {
    if (!s) {
      buf[pos++ & mask] = ArgId_char_nullptr;
      return;
    }
    const uint32_t N = ArgTraitsStrLen(s);
    detail::StringFillData(s, N, buf, pos, mask);
  }
};

template <typename T>
struct ArgTraits;

// template <typename T>
// struct ArgTraits {
//  static const int id = ArgId_None;
//
//  static unsigned int ArgLength(T&& t) {
//    return 4 + sizeof(T);
//  }
//};

template <>
struct ArgTraits<char> {
  static const int id = ArgId_char;

  static __device__ unsigned int ArgLength(const signed char t) {
    return ArgAlign(4);
  }

  static __device__ void Fill(const char ch,
                              uint32_t*  buf,
                              uint32_t&  pos,
                              uint32_t   mask) {
    buf[pos++ & mask] = ch << 16 | id;
  }
};

template <>
struct ArgTraits<signed char> {
  static const int id = ArgId_schar;

  static __device__ unsigned int ArgLength(const signed char t) {
    return ArgAlign(sizeof(t));
  }

  static __device__ void Fill(const signed char ch,
                              uint32_t*         buf,
                              uint32_t&         pos,
                              uint32_t          mask) {
    buf[pos++ & mask] = ch << 16 | id;
  }
};

template <>
struct ArgTraits<unsigned char> {
  static const int id = ArgId_uchar;

  static __device__ unsigned int ArgLength(const unsigned char t) {
    return ArgAlign(sizeof(t));
  }

  static __device__ void Fill(const unsigned char ch,
                              uint32_t*           buf,
                              uint32_t&           pos,
                              uint32_t            mask) {
    buf[pos++ & mask] = ch << 16 | id;
  }
};

template <>
struct ArgTraits<short int> {
  static const int id = ArgId_short;

  static __device__ unsigned int ArgLength(const short int s) {
    return ArgAlign(sizeof(s));
  }

  static __device__ void Fill(const short int ch,
                              uint32_t*       buf,
                              uint32_t&       pos,
                              uint32_t        mask) {
    buf[pos++ & mask] = (ch << 16) | id;
  }
};

template <>
struct ArgTraits<unsigned short int> {
  static const int id = ArgId_ushort;

  static __device__ unsigned int ArgLength(const unsigned short int s) {
    return ArgAlign(sizeof(s));
  }

  static __device__ void Fill(const unsigned short ch,
                              uint32_t*            buf,
                              uint32_t&            pos,
                              uint32_t             mask) {
    buf[pos++ & mask] = (ch << 16) | id;
  }
};

template <>
struct ArgTraits<int> {
  static const int id = ArgId_int;

  static __device__ unsigned int ArgLength(const int i) {
    return 4 + sizeof(i);
  }

  static __device__ void Fill(const int s,
                              uint32_t* buf,
                              uint32_t& pos,
                              uint32_t  mask) {
    detail::FundamentalFill(ArgId_int, s, buf, pos, mask);
  }
};

template <>
struct ArgTraits<unsigned int> {
  static const int id = ArgId_uint;

  static __device__ unsigned int ArgLength(const unsigned int i) {
    return 4 + sizeof(i);
  }

  static __device__ void Fill(const unsigned int s,
                              uint32_t*          buf,
                              uint32_t&          pos,
                              uint32_t           mask) {
    detail::FundamentalFill(ArgId_uint, s, buf, pos, mask);
  }
};

template <>
struct ArgTraits<long> {
  static const long id = ArgId_long;

  static __device__ unsigned int ArgLength(const long l) {
    return 4 + sizeof(l);
  }

  static __device__ void Fill(const long s,
                              uint32_t*  buf,
                              uint32_t&  pos,
                              uint32_t   mask) {
    detail::FundamentalFill(ArgId_long, s, buf, pos, mask);
  }
};

template <>
struct ArgTraits<unsigned long> {
  static const int id = ArgId_ulong;

  static __device__ unsigned int ArgLength(const unsigned long l) {
    return 4 + sizeof(l);
  }

  static __device__ void Fill(const unsigned long s,
                              uint32_t*           buf,
                              uint32_t&           pos,
                              uint32_t            mask) {
    detail::FundamentalFill(ArgId_ulong, s, buf, pos, mask);
  }
};

template <>
struct ArgTraits<long long> {
  static const long id = ArgId_longlong;

  static __device__ unsigned int ArgLength(const long long l) {
    return 4 + sizeof(l);
  }

  static __device__ void Fill(const long long s,
                              uint32_t*       buf,
                              uint32_t&       pos,
                              uint32_t        mask) {
    detail::FundamentalFill(ArgId_longlong, s, buf, pos, mask);
  }
};

template <>
struct ArgTraits<unsigned long long> {
  static const int id = ArgId_ulonglong;

  static __device__ unsigned int ArgLength(const unsigned long long l) {
    return sizeof(l) + 4;
  }

  static __device__ void Fill(const unsigned long long s,
                              uint32_t*                buf,
                              uint32_t&                pos,
                              uint32_t                 mask) {
    detail::FundamentalFill(ArgId_ulonglong, s, buf, pos, mask);
  }
};

template <>
struct ArgTraits<float> {
  static const int id = ArgId_float;

  static __device__ unsigned int ArgLength(const float l) {
    return sizeof(l) + 4;
  }

  static __device__ void Fill(const float s,
                              uint32_t*   buf,
                              uint32_t&   pos,
                              uint32_t    mask) {
    detail::FundamentalFill(ArgId_float, s, buf, pos, mask);
  }
};

template <>
struct ArgTraits<double> {
  static const int id = ArgId_double;

  static __device__ unsigned int ArgLength(const double d) {
    return sizeof(d) + 4;
  }

  static __device__ void Fill(const double s,
                              uint32_t*    buf,
                              uint32_t&    pos,
                              uint32_t     mask) {
    detail::FundamentalFill(ArgId_double, s, buf, pos, mask);
  }
};

template <>
struct ArgTraits<long double> {
  static const int id = ArgId_long_double;

  static __device__ unsigned int ArgLength(const long double d) {
    return sizeof(d) + 4;
  }

  static __device__ void Fill(const long double s,
                              uint32_t*         buf,
                              uint32_t&         pos,
                              uint32_t          mask) {
    detail::FundamentalFill(ArgId_long_double, s, buf, pos, mask);
  }
};

template <>
struct ArgTraits<std::nullptr_t> {
  static const int id = ArgId_nullptr;

  static __device__ unsigned int ArgLength(...) { return 4; }

  static __device__ void Fill(const std::nullptr_t,
                              uint32_t* buf,
                              uint32_t& pos,
                              uint32_t  mask) {
    buf[pos++ & mask] = ArgId_nullptr;
  }
};

template <typename T>
struct ArgTraits<T*> {
  static const int id = ArgId_pointer;

  static __device__ unsigned int ArgLength(...) { return 4 + sizeof(T*); }

  static __device__ void Fill(T* const  s,
                              uint32_t* buf,
                              uint32_t& pos,
                              uint32_t  mask) {
    detail::FundamentalFill(ArgId_pointer, s, buf, pos, mask);
  }
};

template <>
struct ArgTraits<char*> {
  static const int id = ArgId_char_pointer;

  static __device__ unsigned int ArgLength(const char* s) {
    return 4 + (s ? ArgAlign(ArgTraitsStrLen(s)) + 8 : 0);
  }

  static __device__ void Fill(const char* s,
                              uint32_t*   buf,
                              uint32_t&   pos,
                              uint32_t    mask) {
    if (!s) {
      buf[pos++ & mask] = ArgId_char_nullptr;
      return;
    }
    const uint32_t N  = ArgTraitsStrLen(s);
    buf[pos++ & mask] = (N << 16) | id;
    buf[pos++ & mask] = lower_32_bit(s);
    buf[pos++ & mask] = upper_32_bit(s);
    detail::StringFillData(s, N, buf, pos, mask);
  }
};

template <>
struct ArgTraits<const char*> {
  static const int id = ArgId_char_pointer;

  static __device__ unsigned int ArgLength(const char* s) {
    return 4 + (s ? ArgAlign(ArgTraitsStrLen(s)) + 8 : 0);
  }

  static __device__ void Fill(const char* s,
                              uint32_t*   buf,
                              uint32_t&   pos,
                              uint32_t    mask) {
    if (!s) {
      buf[pos++ & mask] = ArgId_char_nullptr;
      return;
    }
    const uint32_t N  = ArgTraitsStrLen(s);
    buf[pos++ & mask] = (N << 16) | id;
    buf[pos++ & mask] = lower_32_bit(s);
    buf[pos++ & mask] = upper_32_bit(s);
    detail::StringFillData(s, N, buf, pos, mask);
  }
};

template <unsigned N>
struct ArgTraits<char[N]> {
  static const int id = ArgId_char_array;

  static __device__ unsigned int ArgLength(const char* s) {
    return 4 + ArgAlign(N) + 8;
  }

  static __device__ void Fill(const char* s,
                              uint32_t*   buf,
                              uint32_t&   pos,
                              uint32_t    mask) {
    buf[pos++ & mask] = (N << 16) | id;
    buf[pos++ & mask] = lower_32_bit(s);
    buf[pos++ & mask] = upper_32_bit(s);
    detail::StringFillData(s, N, buf, pos, mask);
  }
};

template <unsigned N>
struct ArgTraits<signed char[N]> {
  static const int id = ArgId_schar_array;

  static __device__ unsigned int ArgLength(const char* s) {
    return 4 + ArgAlign(N) + 8;
  }

  static __device__ void Fill(const char* s,
                              uint32_t*   buf,
                              uint32_t&   pos,
                              uint32_t    mask) {
    buf[pos++ & mask] = (N << 16) | id;
    buf[pos++ & mask] = lower_32_bit(s);
    buf[pos++ & mask] = upper_32_bit(s);
    detail::StringFillData(s, N, buf, pos, mask);
  }
};

template <unsigned N>
struct ArgTraits<unsigned char[N]> {
  static const int id = ArgId_uchar_array;

  static __device__ unsigned int ArgLength(const char* s) {
    return 4 + ArgAlign(N) + 8;
  }

  static __device__ void Fill(const char* s,
                              uint32_t*   buf,
                              uint32_t&   pos,
                              uint32_t    mask) {
    buf[pos++ & mask] = (N << 16) | id;
    buf[pos++ & mask] = lower_32_bit(s);
    buf[pos++ & mask] = upper_32_bit(s);
    detail::StringFillData(s, N, buf, pos, mask);
  }
};

// template <typename T, unsigned N>
// struct ArgTraits<T[N]> {
//  using type = T;
//
//  static const int id = ArgId_any_array;
//};

// template <unsigned N>
// struct ArgTraits<unsigned char[N]> {
//  static const int id = ArgId_uchar_array;
//};

namespace detail {
template <typename... Args>
__device__ unsigned SumArgsLength(Args&&... args);

template <>
__device__ constexpr unsigned SumArgsLength() {
  return 0;
}

template <typename T, typename... Args>
__device__ unsigned SumArgsLength(T&& t, Args&&... args) {
  typedef typename std::remove_reference<T>::type _type;
  typedef typename std::remove_cv<_type>::type    type;

  return ArgTraits<type>::ArgLength(std::forward<T>(t)) +
         SumArgsLength(std::forward<Args>(args)...);
}

struct FifoInfo {
  uint32_t get;
  uint32_t put;

  //!< num words
  uint32_t size;

  uint32_t fifoSize;
  uint64_t fifoAddress;
};

template <typename T>
struct ArgTraitsHasFillFifoInfo {
  template <typename U>
  static auto Check(int) -> decltype(&U::FillFifoInfo);

  template <typename U>
  static void Check(...);

  static const bool value = !std::is_same<decltype(Check<T>(0)), void>::value;
};

template <typename AT, bool value>
struct ArgTraitsFillProxy;

template <typename T>
struct ArgTraitsFillProxy<T, true> {
  template <typename U>
  __device__ static void Fill(U&&             u,
                              uint32_t*       fifobuf,
                              const FifoInfo& msgInfo,
                              uint32_t&       pos,
                              uint32_t&       mask) {
    ArgTraits<T>::FillFifoInfo(std::forward<U>(u), fifobuf, msgInfo, pos, mask);
  }
};

template <typename T>
struct ArgTraitsFillProxy<T, false> {
  template <typename U>
  __device__ static void Fill(U&&             u,
                              uint32_t*       fifobuf,
                              const FifoInfo& msgInfo,
                              uint32_t&       pos,
                              uint32_t&       mask) {
    ArgTraits<T>::Fill(std::forward<U>(u), fifobuf, pos, mask);
  }
};

template <typename... Args>
__device__ void FillArgs(uint32_t*       fifobuf,
                         const FifoInfo& msgInfo,
                         uint32_t&       pos,
                         uint32_t        mask,
                         Args&&... args);

template <>
__device__ inline void FillArgs(uint32_t*       fifobuf,
                                const FifoInfo& msgInfo,
                                uint32_t&       pos,
                                uint32_t        mask) {}

template <typename T, typename... Args>
__device__ void FillArgs(uint32_t*      fifobuf,
                         const FifoInfo& msgInfo,
                         uint32_t&      pos,
                         uint32_t       mask,
                         T&&            t,
                         Args&&... args) {
  typedef typename std::remove_reference<T>::type _type;
  typedef typename std::remove_cv<_type>::type    type;

  ArgTraitsFillProxy<type, ArgTraitsHasFillFifoInfo<ArgTraits<type>>::value>::
    Fill(t, fifobuf, msgInfo, pos, mask);
  //ArgTraits<type>::Fill(t, fifobuf, pos, mask);
  FillArgs(fifobuf, msgInfo, pos, mask, std::forward<Args>(args)...);
}

template <typename... Args>
struct CountOfArgs;

template <>
struct CountOfArgs<> {
  static const int value = 0;
};

template <typename T, typename... Args>
struct CountOfArgs<T, Args...> {
  static const int value = CountOfArgs<Args...>::value + 1;
};

}  // namespace detail

namespace debug {

//!< The get the __pt_printf load.
struct MsgGet {};

//!< The begin position of the current __pt_printf.
struct MsgBeg {};

//!< The number words the current __pt_printf will consume.
struct MsgSize {};

//!< Print the begin address of the current __pt_printf fifo.
struct FifoAddress {};

//!< Print the size of the print fifo.
struct FifoSize {};

}  // namespace debug

template <>
struct ArgTraits<debug::MsgBeg> {
  static const int id = ArgId_uint;

  __device__ static unsigned int ArgLength(debug::MsgBeg put) { return 8; }

  __device__ static void FillFifoInfo(debug::MsgBeg,
                                      uint32_t*               fifobuf,
                                      const detail::FifoInfo& msgInfo,
                                      uint32_t&               pos,
                                      uint32_t                mask) {
    detail::FundamentalFill(ArgId_uint, msgInfo.put, fifobuf, pos, mask);
  }
};

template <>
struct ArgTraits<debug::MsgGet> {
  static const int id = ArgId_uint;

  __device__ static unsigned int ArgLength(debug::MsgGet put) { return 8; }

  __device__ static void FillFifoInfo(debug::MsgGet,
                                      uint32_t*               fifobuf,
                                      const detail::FifoInfo& msgInfo,
                                      uint32_t&               pos,
                                      uint32_t                mask) {
    detail::FundamentalFill(ArgId_uint, msgInfo.get, fifobuf, pos, mask);
  }
};

template <>
struct ArgTraits<debug::MsgSize> {
  static const int id = ArgId_uint;

  __device__ static unsigned int ArgLength(debug::MsgSize s) { return 8; }

  __device__ static void FillFifoInfo(debug::MsgSize,
                                      uint32_t*               fifobuf,
                                      const detail::FifoInfo& msgInfo,
                                      uint32_t&               pos,
                                      uint32_t                mask) {
    detail::FundamentalFill<uint32_t>(ArgId_uint,
                                      msgInfo.size,
                                      fifobuf,
                                      pos,
                                      mask);
  }
};

template <>
struct ArgTraits<debug::FifoSize> {
  static const int id = ArgId_uint;

  __device__ static unsigned int ArgLength(debug::FifoSize s) { return 8; }

  __device__ static void FillFifoInfo(debug::FifoSize,
                                      uint32_t*               fifobuf,
                                      const detail::FifoInfo& msgInfo,
                                      uint32_t&               pos,
                                      uint32_t                mask) {
    detail::FundamentalFill<uint32_t>(ArgId_uint,
                                      msgInfo.fifoSize,
                                      fifobuf,
                                      pos,
                                      mask);
  }
};

template <>
struct ArgTraits<debug::FifoAddress> {
  static const int id = ArgId_pointer;

  __device__ static unsigned int ArgLength(debug::FifoAddress s) { return 12; }

  __device__ static void FillFifoInfo(debug::FifoAddress,
                                      uint32_t*               fifobuf,
                                      const detail::FifoInfo& msgInfo,
                                      uint32_t&               pos,
                                      uint32_t                mask) {
    detail::FundamentalFill<uint64_t>(ArgId_pointer,
                                      msgInfo.fifoAddress,
                                      fifobuf,
                                      pos,
                                      mask);
  }
};

struct fifo {
  unsigned int put __attribute__((aligned(128)));
  unsigned int mask;
  uint32_t*    data;

  unsigned int ready __attribute__((aligned(128)));

  unsigned int get __attribute__((aligned(128)));
};

extern "C" struct fifo __ptPrintfFifo;

inline __device__ struct fifo* __ptSelectPrintfFifo(void) {
  //unsigned int bidx = threadIdx.z * (blockDim.x * blockDim.y) +
  //                    threadIdx.y * blockDim.x + threadIdx.x;
  //return &__ptPrintfFifo + (widx / 32) & 0x01;
#ifdef __TANGC_MAJOR__
  return &__ptPrintfFifo + (__phywarpid() & 0x01);
#else
  return &__ptPrintfFifo;
#endif  //!< __TANGC_MAJOR__
}

inline __device__ uint32_t __ptPrintfFifoAlloc(struct fifo* fifo,
                                               uint32_t     n,
                                               uint32_t*    pGet) {
  [[maybe_unused]] unsigned int tmp;
  unsigned int newPut;
  unsigned int oldPut;
  unsigned int avail;
  unsigned int get;
  unsigned int mask = fifo->mask;

  if (n >= mask) {
    return std::numeric_limits<uint32_t>::max();
  }
  do {
#ifdef __TANGC_MAJOR__
    oldPut = *((volatile unsigned int*)&fifo->put);
    // oldPut = __ldcg(&fifo->put);
#else
    oldPut = __atomic_load_n(&fifo->put, __ATOMIC_RELAXED);
#endif  //!< __TANGC_MAJOR__

//#ifdef __TANGC_MAJOR__
//    __threadfence_memory();
//#endif  //!< __TANGC_MAJOR__

#ifdef __TANGC_MAJOR__
    // get = *((volatile unsigned int*)&fifo->get);
    get = __ldcg(&fifo->get);
#else
    get = __atomic_load_n(&fifo->get, __ATOMIC_RELAXED);
#endif  //!< __TANGC_MAJOR__

    // avail = mask - ((oldPut - get) & mask);
    avail = (get - oldPut - 1) & mask;
    if (avail < n) {
      return std::numeric_limits<uint32_t>::max();
    }
    newPut = oldPut + n;
    // newPut = (oldPut + n) & mask;
#ifdef __TANGC_MAJOR__
    tmp = atomicCAS(&fifo->put, oldPut, newPut);
    // Compiler group provides this solution.
#  if 1
    asm volatile("loop 1, 0, 1, 500000\n\tnop");
#  else
    __stvm_bar_sync0();
#  endif
  } while (oldPut != tmp);
#else
  } while (!__atomic_compare_exchange_n(&fifo->put,
                                        &oldPut,
                                        newPut,
                                        true,
                                        __ATOMIC_RELAXED,
                                        __ATOMIC_RELAXED));
#endif  //!< __TANGC_MAJOR__

  *pGet = get;
  return oldPut;
}

inline __device__ void __ptPrintfFifoUpdateReady(struct fifo* fifo,
                                                 uint32_t     orig_pos,
                                                 uint32_t     pos) {
  [[maybe_unused]] unsigned int tmp;
  unsigned int oldReady = orig_pos;
#ifdef __TANGC_MAJOR__
  //! Make sure all writes before this call happens before
  //! all writes after this call.
  //! __syncthreads();
  __threadfence_memory();
#endif  //!< __TANGC_MAJOR__
  do {
#ifdef __TANGC_MAJOR__
    tmp = atomicCAS(&fifo->ready, oldReady, pos);
#  if 1
    asm volatile("loop 1, 0, 1, 500000\n\tnop");
#  else
    __stvm_bar_sync0();
#  endif
  } while (oldReady != tmp);
#else
  } while (!__atomic_compare_exchange_n(&fifo->ready,
                                        &oldReady,
                                        pos,
                                        true,
                                        __ATOMIC_RELEASE,
                                        __ATOMIC_RELAXED));
#endif  //!< __TANGC_MAJOR__
}

template <typename Fmt, typename... Args>
__device__ void __pt_printf(Fmt&& fmt, Args&&... args) {
  typedef typename std::remove_reference<Fmt>::type _fmt_type;
  typedef typename std::remove_cv<_fmt_type>::type  fmt_type;

  auto fifo = __ptSelectPrintfFifo();

  // struct fifo = __ptPrintfFifo;
  // numWords:     the number bytes of the message;
  // countOfArgs:  the number of args
  // fmt:          fmt data
  // arg[numArgs]: arg data
  // endMarker:    maybe not required

  unsigned int numFmtWords = FmtTraits<fmt_type>::FmtLength(fmt) / 4;
  unsigned int numArgWords =
    detail::SumArgsLength(std::forward<Args>(args)...) / 4;

  static_assert(detail::CountOfArgs<Args...>::value == sizeof...(Args), "");

  // unsigned int countOfArgs = detail::CountOfArgs<Args...>::value;
  unsigned int countOfArgs = sizeof...(Args);

#ifdef PT_PRINTF_ENDMARKER
  unsigned int numMsgWords = 3 + numFmtWords + numArgWords;
#else
  unsigned int numMsgWords = 2 + numFmtWords + numArgWords;
#endif  //!< PT_PRINTF_ENDMARKER

  // align message to 128byte, 32uint32_t boundary.
#if 0
  numMsgWords = (numMsgWords + 31) & ~31;
#endif

  uint32_t get;
  uint32_t pos = __ptPrintfFifoAlloc(fifo, numMsgWords, &get);
  if (pos == std::numeric_limits<uint32_t>::max()) {
    return;
  }

  const detail::FifoInfo fifoInfo = {
    .get         = get,
    .put         = pos,
    .size        = numMsgWords,
    .fifoSize    = fifo->mask + 1,
    .fifoAddress = (uint64_t)fifo->data,
  };

  uint32_t const  orig_pos = pos;
  uint32_t const  mask     = fifo->mask;
  uint32_t* const fifobuf  = (uint32_t*)fifo->data;

  fifobuf[pos++ & mask] = numMsgWords;
  fifobuf[pos++ & mask] = countOfArgs;

  FmtTraits<fmt_type>::Fill(std::forward<Fmt>(fmt), fifobuf, pos, mask);
  detail::FillArgs(fifobuf, fifoInfo, pos, mask, std::forward<Args>(args)...);

#ifdef PT_PRINTF_ENDMARKER
  fifobuf[pos++ & mask] = numMsgWords;
#endif  //!< PT_PRINTF_ENDMARKER
#if 0
  while ((pos - orig_pos) < numMsgWords) {
    fifobuf[pos++ & mask] = orig_pos;
  }
#endif

  __ptPrintfFifoUpdateReady(fifo, orig_pos, (orig_pos + numMsgWords));
}

}  // namespace fmt
}  // namespace tangrt

using tangrt::fmt::__pt_printf;

#endif  //!< _TANGRT_FMT_HPP_
