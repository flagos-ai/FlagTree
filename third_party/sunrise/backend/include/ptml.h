////////////////////////////////////////////////////////
// @file ptml.h
// ptml api
// ptmlDevice_t represents the type of device index
////////////////////////////////////////////////////////

#ifndef _PT_ML_H_
#define _PT_ML_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif  //! __cplusplus

#include <string.h>

#ifndef TA_PT_NUM_MAX
#define TA_PT_NUM_MAX 128
#endif  //! TA_PT_NUM_MAX

#if defined(_MSC_VER)
#define PTML_DEPRECATED __declspec(deprecated)
#define PTML_API_EXPORT __declspec(dllexport)
#define PTML_API_IMPORT __declspec(dllimport)
#elif defined(__GNUC__) || defined(__clang__)
#define PTML_DEPRECATED __attribute__((deprecated))
#define PTML_API_EXPORT __attribute__((visibility("default")))
#define PTML_API_IMPORT __attribute__((visibility("default")))
#else
#define PTML_DEPRECATED
#define PTML_API_EXPORT
#define PTML_API_IMPORT
#endif  //! UNKNOWN COMPILER

#if defined(ptml_EXPORTS)
#define PTML_API PTML_API_EXPORT
#else
#define PTML_API PTML_API_IMPORT
#endif  //! For user

#define PAGE_SIZE                    4096
#ifndef ALIGN
#define __ALIGN_KERNEL_MASK(x, mask) (((x) + (mask)) & ~(mask))
#define __ALIGN_KERNEL(x, a)         __ALIGN_KERNEL_MASK(x, (__typeof__(x))(a)-1)
#define ALIGN(x, a)                  __ALIGN_KERNEL((x), (a))
#endif //  ALIGN
#define PAGE_ALIGN(addr)             ALIGN(addr, PAGE_SIZE)
#define CLK_NAME_MAX                 16
#define NUMBER_OF_CYCLES_IN_1_SEC    0x38400000       // 900M
#define am_interval_1s               (1800000000)  // 1800M stands for 1 second

/*ioctl parm cmd*/
#define CM3   0x10
#define MLP   0x11
#define LINUX 0x12

#define CMD_PMIC          (0xa8)
#define MOD_TEMP_TYPE_NUM (1)
#define TEMP_IPID_CNT     (8)
#define C2C_INFO_CNT      (10)

#define CMD_TEMPERATURE 0xab
#define CMD_HBM_TEMPERATURE 0xb2
#define CMD_GET_GPIO_STATUS 0xb7
#define CMD_DUMP_MEM        0xb5
#define CMD_GET_CPLD_VERSION 0xbc
#define CMD_GET_MAX_POWER    0xbd
#define CMD_GET_EXCEPTION    0xb6

/*linux cmd*/
#define LINUX_CMD_PTUTILI     (0x11)
#define LINUX_CMD_PCIERELINK  (0x13)
#define LINUX_CMD_HBMBWUTILI  (0x1A)
#define LINUX_CMD_HBMUTILI    (0x1B)
#define LINUX_CMD_C2CREVDB    (0x14)
#define LINUX_CMD_C2CTRANSDB  (0x15)
#define LINUX_CMD_PCIEREVDB   (0x16)
#define LINUX_CMD_PCIETRANSDB (0x17)
#define LINUX_CMD_TUUTILI     (0x18)
#define LINUX_CMD_THREADUTILI (0x19)

/**
 * @brief Return val for ptml API
 */
typedef enum ptmlReturn_enum {
  PTML_SUCCESS = 0,                //!< APT returns ok
  PTML_ERROR_UNINITIALIZED,        //!< ptmlInit is not called now
  PTML_ERROR_INVALID_ARGUMENT,     //!< invalid argument
  PTML_ERROR_ALREADY_INITIALIZED,  //!< ptmlInit is already called
  PTML_ERROR_INSUFFICIENT_SIZE,    //!< An input argument is not large enough
  PTML_ERROR_IN_USE,               //!< PT is in use
  PTML_ERROR_DRIVER_NOT_LOADED,    //!< driver is not loaded
  PTML_ERROR_DEVICE_NOT_FOUND,     //!< device is not found
  PTML_ERROR_EVENT_TIMEOUT,        //!< device is not found
  PTML_ERROR_UNKNOWN,              //!< An internal driver error occurred
} ptmlReturn_t;
  

typedef struct {
  enum ptmlReturn_enum errorCode;
  const char          *errorMessage;
} ErrorDescriptor;

typedef enum ptmlClockType {
  //!< PTML_CLOCK_GRAPHICS = 0,
  //!< PTML_CLOCK_SM,
  PTML_CLOCK_PT = 0,
  PTML_CLOCK_MEM,
  //!< PTML_CLOCK_VIDEO,
} ptmlClockType_t;

/**
 * @brief Device Handle type
 *
 ********************************************/
typedef int ptmlDevice_t;

typedef struct ptMemory {
  size_t total;  //!< total memory
  size_t used;   //!< used memory
  size_t free;   //!< free memory
} ptMemory_t;

#define PTML_DEVICE_PCI_BUS_ID_BUFFER_SIZE 32

typedef struct ptPciInfo {
  int  domain;                                     //!< domain number
  int  bus;                                        //!< bus number
  int  device;                                     //!< dev && func number
  int  vendor;                                     //!< vendor number
  int  pciSubSystemId;                             //!< subsystem Id
  char busId[PTML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];  //!< "domain:bus:device:0"
  unsigned int max_link_speed;                     //!< max link speed(MT/s)
  unsigned int max_link_width;                     //!< max link width
  unsigned int max_bandwidth;                      //!< max bandwidth(MB/s)
  unsigned int curr_link_speed;                    //!< current link speed(MT/s)
  unsigned int curr_link_width;                    //!< current link width
  unsigned int curr_bandwidth;                     //!< current bandwidth(MB/s)
} ptPciInfo_t;

typedef struct ptProcessInfo {
  char name[256];
  char pidStr[16];
}ptProcessInfo_t;

struct barInfo {
  int      barIdx;  //!< 0-5
  void*    addr;    //!< The mapped vritual address.
  uint64_t paddr;
  size_t   size;  //!< The size of the mapped vritaul address space.
};

struct fw_param {
  int cmd;
  int len;
  int data[];
};
#define _S2_IOC_SMI_INFO _IOWR(_S2_IOC_MAGIC, 160, struct fw_param)

struct st_rpmsg_cmd {
  unsigned int  cmd;
  unsigned char data[];
} __attribute__((packed));

struct st_rpmsg_i2c_cmd {
  unsigned char rx_data_len;
  unsigned char tx_data_len;
  unsigned char slave_len;
  unsigned char trx_flag;
  unsigned char pyload[];  // slave addr + tx data
} __attribute__((packed));

struct st_rpmsg_i2c_response {
  unsigned char error_code;
  unsigned char rx_data_len;
  unsigned char slave_len;
  unsigned char trx_flag;
  unsigned char pyload[];  // slave addr + rx data
} __attribute__((packed));

struct st_rpmsg_pmic_cmd {
  unsigned char pmic_cmd;
  unsigned int  pmic_param;
} __attribute__((packed));

struct st_rpmsg_pmic_response {
  unsigned char pmic_cmd;
  unsigned char error_code;
  unsigned char pmic_data[4];
} __attribute__((packed));

/*
 * RPmsg Clock Command IDs
 */
enum rpmsg_clk_cmd_id {
  RPMSG_CLK_GET_STATE,
  RPMSG_CLK_GET_NAME,
  RPMSG_CLK_GET_RATE,
  RPMSG_CLK_ENABLE,
  RPMSG_CLK_DISABLE,
  RPMSG_CLK_CMD_COUNT,
};

enum plat_clock_idx {
  MOD_CLOCK_G0_0,
  MOD_CLOCK_G0_1,
  MOD_CLOCK_G1_0,
  MOD_CLOCK_G2_0,
  MOD_CLOCK_G3_0,
  MOD_CLOCK_G4_0,
  MOD_CLOCK_G5_0,
  MOD_CLOCK_G6_0,
  MOD_CLOCK_G7_0,
  MOD_CLOCK_G8_0,
  MOD_CLOCK_G9_0,
  MOD_CLOCK_G10_0,
  MOD_CLOCK_G11_0,
  MOD_CLOCK_G12_0,
  MOD_CLOCK_G13_0,
  MOD_CLOCK_G13_1,
  MOD_CLOCK_G13_2,
  MOD_CLOCK_G14_0,
  MOD_CLOCK_G14_1,

  MOD_CLOCK_L0_CLK2000CLK,
  MOD_CLOCK_L0_CLK1000CLK,
  MOD_CLOCK_L0_CLK500CLK,
  MOD_CLOCK_L0_CLK250CLK,
  MOD_CLOCK_L0_CLK125CLK,
  MOD_CLOCK_L0_CLK62P5CLK,
  MOD_CLOCK_L0_CLK31P25CLK,
  MOD_CLOCK_L0_SMB_MELESCLK,
  MOD_CLOCK_L0_MELS_REF_CLK,
  MOD_CLOCK_L0_SMB_32KCLK,
  MOD_CLOCK_L0_PLL0_CLK,
  MOD_CLOCK_L1_CORE_CLK_L,
  MOD_CLOCK_L1_NOC_CLK0,
  MOD_CLOCK_L1_PLL1_CLK,
  MOD_CLOCK_L2_CORE_CLK_H,
  MOD_CLOCK_L2_NOC_CLK1,
  MOD_CLOCK_L2_PLL2_CLK,
  MOD_CLOCK_L3_APBCLK,
  MOD_CLOCK_L3_PLL3_CLK,
  MOD_CLOCK_L4_VIDEO_CLK,
  MOD_CLOCK_L4_PLL4_CLK,
  MOD_CLOCK_L5_DMA_CLK,
  MOD_CLOCK_L5_TIGER_CLK,
  MOD_CLOCK_L5_PLL5_CLK,
  MOD_CLOCK_L6_AXI_CLOCK0,
  MOD_CLOCK_L6_PLL6_CLK,
  MOD_CLOCK_L7_AXI_CLOCK1,
  MOD_CLOCK_L7_PLL7_CLK,
  MOD_CLOCK_L8_JPEG_CLK,
  MOD_CLOCK_L8_PLL8_CLK,
  MOD_CLOCK_L9_PLLREFCLK0,
  MOD_CLOCK_L9_DFICLK0,
  MOD_CLOCK_L9_DFIHDRCLK0,
  MOD_CLOCK_L9_PLL9_CLK,
  MOD_CLOCK_L10_PLLREFCLK1,
  MOD_CLOCK_L10_DFICLK1,
  MOD_CLOCK_L10_DFIHDRCLK1,
  MOD_CLOCK_L10_PLL10_CLK,
  MOD_CLOCK_L11_PLLREFCLK2,
  MOD_CLOCK_L11_DFICLK2,
  MOD_CLOCK_L11_DFIHDRCLK2,
  MOD_CLOCK_L11_PLL11_CLK,
  MOD_CLOCK_L12_PLLREFCLK3,
  MOD_CLOCK_L12_DFICLK3,
  MOD_CLOCK_L12_DFIHDRCLK3,
  MOD_CLOCK_L12_PLL12_CLK,
  MOD_CLOCK_L13_ACLK0,
  MOD_CLOCK_L13_PLL13_CLK,
  MOD_CLOCK_L15_AUX_CLK0,
  MOD_CLOCK_L16_ACLK1,
  MOD_CLOCK_L16_PLL16_CLK,
  MOD_CLOCK_L17_AUX_CLK1,

  MOD_CLOCK_IDX_COUNT,
};

enum c2c_port_index {
  C2C0_0 = 0,
  C2C0_1,
  C2C1_0,
  C2C1_1,
  C2C2_0,
  C2C2_1,
  C2C3_0,
  C2C3_1,
  C2C4_0,
  C2C4_1,
  PCIE,
};
/*
 * struct c2h_clk_msg - Response payload for RPMSG_CLK_ATTRIBUTES_DUMP command
 * @status: Command status
 * @state:  Clock state(on or off)
 * @rate:   Clock rate in Hz,
 *  rate[0] 32bit lsb clock rate
 *  rate[1] 32bit hsb clock rate
 * @name:   Clock name
 */
struct c2h_clk_msg {
  int      status;
  uint32_t state;
  uint32_t rate[2];
  char     name[16];
};

struct c2h_temp_msg {
  int status;
  int temp[TEMP_IPID_CNT];
};

#define ALLPORTS 0x3ff
typedef struct ptPhyTopo {
  unsigned char local_chipid;
  unsigned char local_port;
  unsigned char remote_chipid;
  unsigned char remote_port;
  unsigned char link_status;
  unsigned char isBif;
  unsigned char max_speed;
  unsigned char cur_speed;
  unsigned char max_bandwidth;
  unsigned char cur_bandwidth;
} ptPhyTopo_t;

/**
 * @brief Init ptml module
 *
 * @return int
 * @note must called before using any ptml API
 ********************************************/
ptmlReturn_t PTML_API ptmlInit(void);
ptmlReturn_t PTML_API __ptmlUinit(void);
void PTML_API         ptmlUninit(void);

/**
 * @brief Get system driver version 0.1.0
 *
 * @param length is driver version's length
 * @note  driver Version is Equivalent to dirver version, specific information
 *in
 *  "/sys/module/pt/version"
 * @return ptmlReturn_t
 ********************************************/
ptmlReturn_t PTML_API ptmlSystemGetDriverVersion(char *       version,
                                                 unsigned int length);

/**
 * @brief Get system tang version 0.1.0
 *
 * @note  tang Version is Equivalent to cuda version
 * @return ptmlReturn_t
 ********************************************/
ptmlReturn_t PTML_API ptmlSystemGetTangVersion(int *version);
ptmlReturn_t PTML_API ptmlSystemGetTangVersionForSmi(char *       version,
                                                     unsigned int length);

/**
 * @brief Get dev base info: ptType and memInfo
 *
 * @param device device handles
 * @param ptTypeOut pointer to ptTypeOut
 * @param memInfo pointer to dev memInfo
 * @note  ptTypeOut is pt200, device info is total mem size
 * @return ptmlReturn_t
 ********************************************/
ptmlReturn_t PTML_API ptmlDeviceGetBaseInfo(ptmlDevice_t  device,
                                            char *        ptTypeOut,
                                            unsigned int *memInfo);

/**
 * @brief Get PT board count
 *
 * @param devCount pointer to devCount
 * @return ptmlReturn_t
 ********************************************/
ptmlReturn_t PTML_API ptmlDeviceGetBoardCount(unsigned int *devCount);

/**
 * @brief Get PT type :pt200
 *
 * @param device device handles
 * @param ptTypeOut pointer to pt type
 * @return ptmlReturn_t
 ********************************************/
ptmlReturn_t PTML_API ptmlDeviceGetBrand(ptmlDevice_t device, char *ptTypeOut);

/**
 * @brief Get device Capacity
 *
 * @param device device handles
 * @param value pointer to Capacity
 * @return ptmlReturn_t
 ********************************************/
ptmlReturn_t PTML_API ptmlDeviceGetCapacity(ptmlDevice_t device, float *value);

/**
 * @brief Get PT count
 *
 * @param devCount pointer to devCcount
 * @return ptmlReturn_t
 ********************************************/
ptmlReturn_t PTML_API ptmlDeviceGetCount(int *devCount);

/**
 * @brief Get device c2c rev DB
 *
 * @param device device handles
 * @param c2cIndex is port num 0-10
 * @param interval is from 1-60(s)
 * @param revdb pointer is c2c revdb
 * @return ptmlReturn_t
 ********************************************/
ptmlReturn_t PTML_API ptmlDeviceGetC2CRevDB(ptmlDevice_t  device,
                                            int           c2cIndex,
                                            int           interval,
                                            unsigned int *revdb);

/**
 * @brief Get device c2c trans DB
 *
 * @param device device handles
 * @param c2cIndex is port num 0-10
 * @param interval is from 1-60(s)
 * @param transdb pointer is c2c transdb
 * @return ptmlReturn_t
 ********************************************/
ptmlReturn_t PTML_API ptmlDeviceGetC2CTransDB(ptmlDevice_t  device,
                                              int           c2cIndex,
                                              int           interval,
                                              unsigned int *transdb);

/**
 * @brief Get device fw version
 *
 * @param device device handles
 * @param version pointer to fw version
 * @param length is version's length
 * @note  fw is cm3 linux mlp mix version Internal use only Not provided
 *externally
 * @return ptmlReturn_t
 ********************************************/
ptmlReturn_t PTML_API ptmlDeviceGetFirmwareVersion(ptmlDevice_t device,
                                                   char *       version,
                                                   unsigned int length);

/**
 * @brief Get Device Handle by idx
 *
 * @param idx idx of the target PT
 * @param device pointer to the handle of target PT
 * @return ptmlReturn_t
 ********************************************/
ptmlReturn_t PTML_API ptmlDeviceGetHanldeByIdx(unsigned int  idx,
                                               ptmlDevice_t *device);

/**
 * @brief Get Device Handle by PCI
 *
 * @param idx idx of the target PT
 * @param device pointer to the handle of target PT
 * @return ptmlReturn_t
 ********************************************/
ptmlReturn_t PTML_API ptmlDeviceGetHanldeByPciBusId(const char *  busId,
                                                    ptmlDevice_t *device);

/**
 * @brief Get device mem BW Utilization
 *
 * @param device device handles
 * @param interval is from 1-60(s)
 * @param utilization pointer is memBWUtilization 0-100%
 * @return ptmlReturn_t
 ********************************************/
ptmlReturn_t PTML_API ptmlDeviceGetMemBWUtilizationRates(ptmlDevice_t device,
                                                         int          interval,
                                                         float *utilization);

/**
 * @brief ptmlDeviceGetMemClockFrequency
 *
 * @param device device handles
 * @param clock pointer to memClockFreq
 * @return ptmlReturn_t
 ********************************************/
ptmlReturn_t PTML_API ptmlDeviceGetMemClockFrequency(ptmlDevice_t  device,
                                                     unsigned int *clock);

/**
 * @brief Get device memory information
 *
 * @param device device handle
 * @param memInfo pointer to memory information
 * @return ptmlReturn_t
 ********************************************/
ptmlReturn_t PTML_API ptmlDeviceGetMemoryInfo(ptmlDevice_t device,
                                              ptMemory_t * memInfo);
ptmlReturn_t PTML_API ptmlDeviceGetMemoryUsedInfo(ptmlDevice_t  device,
                                                  unsigned int *usedInfo);
ptmlReturn_t PTML_API ptmlDeviceGetMemoryFreeInfo(ptmlDevice_t  device,
                                                  unsigned int *usedInfo);

/**
 * @brief Get device mem Temperature
 *
 * @param device device handles
 * @param temp pointer to ptTemperature
 * @return ptmlReturn_t
 ********************************************/
ptmlReturn_t PTML_API ptmlDeviceGetMemTemperature(ptmlDevice_t  device,
                                                  unsigned int *temp);

/**
 * @brief Get device mem Utilization
 *
 * @param device device handles
 * @param interval is from 1-60(s)
 * @param utilization pointer is memUtilization 0-100%
 * @return ptmlReturn_t
 ********************************************/
ptmlReturn_t ptmlDeviceGetMemUtilizationRates(ptmlDevice_t device,
                                                       int          interval,
                                                       float *utilization);

/**
 * @brief Get PT node path /dev/ptpux
 *
 * @param device device handle
 * @param path pointer to devNodePath
 * @return ptmlReturn_t
 ********************************************/
ptmlReturn_t PTML_API ptmlDeviceGetNodePath(ptmlDevice_t device, char *path);

/**
 * @brief Get device pcie relink times
 *
 * @param device device handles
 * @param poniter to pcie relink times
 * @return ptmlReturn_t
 ********************************************/
ptmlReturn_t PTML_API ptmlDeviceGetPcieRelinkTime(ptmlDevice_t device,
                                                  int *        count);

/**
 * @brief Get device pcie rev DB
 *
 * @param device device handles
 * @param interval is from 1-60(s)
 * @param revdb pointer is pcie revdb
 * @return ptmlReturn_t
 ********************************************/
ptmlReturn_t PTML_API ptmlDeviceGetPcieRevDB(ptmlDevice_t  device,
                                             int           interval,
                                             unsigned int *revdb);

/**
 * @brief Get device pcie trans DB
 *
 * @param device device handles
 * @param interval is from 1-60(s)
 * @param transdb pointer is pcie transdb
 * @return ptmlReturn_t
 ********************************************/
ptmlReturn_t PTML_API ptmlDeviceGetPcieTransDB(ptmlDevice_t  device,
                                               int           interval,
                                               unsigned int *transdb);

/**
 * @brief Get device pci information
 *
 * @param device device handle
 * @param pciInfo pointer to pci information
 * @return ptmlReturn_t
 ********************************************/
ptmlReturn_t PTML_API ptmlDeviceGetPciInfo(ptmlDevice_t device,
                                           ptPciInfo_t *pciInfo);

/**
 * @brief Reboot device
 *
 * @param device device handles
 * @return ptmlReturn_t
 ********************************************/
ptmlReturn_t PTML_API ptmlDeviceRebootBootloader(ptmlDevice_t device);

/**
 * @brief Get PT status 0:invalid 1:valid
 *
 * @param device device handles
 * @param status pointer to ptstatus
 * @return ptmlReturn_t
 ********************************************/
ptmlReturn_t PTML_API ptmlDeviceGetStatus(ptmlDevice_t device, int *status);

/**
 * @brief ptmlDeviceGetPtClockFrequency
 *
 * @param device device handles
 * @param clock pointer to ptClockFreq
 * @return ptmlReturn_t
 ********************************************/
ptmlReturn_t PTML_API ptmlDeviceGetPtClockFrequency(ptmlDevice_t  device,
                                                      unsigned int *clock);

/**
 * @brief Get PTCTRL major and minor
 *
 * @param device device handles
 * @param major pointer to ptpuctrl major
 * @param minor pointer to ptpuctrl minor
 * @return ptmlReturn_t
 ********************************************/
ptmlReturn_t PTML_API ptmlDeviceGetPtCtrlMajorAndMinor(int *major,
                                                         int *minor);

/**
 * @brief Get PT major and minor  /dev/ptpux
 *
 * @param device device handles
 * @param major pointer to devmajor
 * @param minor pointer to devminor
 * @return ptmlReturn_t
 ********************************************/
ptmlReturn_t PTML_API ptmlDeviceGetPtMajorAndMinor(ptmlDevice_t device,
                                                     int *        major,
                                                     int *        minor);

/**
 * @brief Get device pt Temperature
 *
 * @param device device handles
 * @param temp pointer to ptTemperature
 * @return ptmlReturn_t
 ********************************************/
ptmlReturn_t PTML_API ptmlDeviceGetPtTemperature(ptmlDevice_t  device,
                                                   unsigned int *temp);

/**
 * @brief Get device pt Utilization
 *
 * @param device device handles
 * @param interval is from 1-60(s)
 * @param utilization pointer to ptUtilization 0-100%
 * @return ptmlReturn_t
 ********************************************/
ptmlReturn_t PTML_API ptmlDeviceGetPtUtilizationRates(ptmlDevice_t device,
                                                        int          interval,
                                                        float *utilization);

/**
 * @brief Get compute capability
 *
 * @param device device handles
 * @param major pointer to major
 * @param minor pointer to minor
 * @return ptmlReturn_t
 ********************************************/
ptmlReturn_t PTML_API ptmlDeviceGetTangComputeCapability(ptmlDevice_t device,
                                                         int *        major,
                                                         int *        minor);

/**
 * @brief Get device thread Utilization
 *
 * @param device device handles
 * @param interval is from 1-60(s)
 * @param utilization pointer to threadUtilization 0-100%
 * @return ptmlReturn_t
 ********************************************/
ptmlReturn_t PTML_API ptmlDeviceGetThreadUtilizationRates(ptmlDevice_t device,
                                                          int          interval,
                                                          float *utilization);

/**
 * @brief Get device subcore Utilization
 *
 * @param device device handles
 * @param interval is from 1-60(s)
 * @param utilization pointer to subcoreUtilization 0-100%
 * @return ptmlReturn_t
 ********************************************/
ptmlReturn_t PTML_API ptmlDeviceGetTUUtilizationRates(ptmlDevice_t device,
                                                      int          interval,
                                                      float *      utilization);

/**
 * @brief Get device uuid
 *
 * @param device device handles
 * @param uuid pointer to device uuid
 * @param length device uuid length
 * @return ptmlReturn_t
 ********************************************/
ptmlReturn_t PTML_API ptmlDeviceGetUUID(ptmlDevice_t device,
                                        char *       uuid,
                                        unsigned int length);

/**
 * @brief ptmlDeviceSetCUFrequency
 *
 * @param device device handles
 * @param CU freq
 * @return ptmlReturn_t
 ********************************************/
ptmlReturn_t PTML_API ptmlDeviceSetCUFrequency(ptmlDevice_t        device,
                                             unsigned int freq) ;

/**
 * @brief Get device Mem Temperature
 *
 * @param device device handles
 * @param temp pointer to ptTemperature
 * @return ptmlReturn_t
 ********************************************/
ptmlReturn_t PTML_API ptmlDeviceGetMemTemperature(ptmlDevice_t  device,
                                                   unsigned int *temp);

/**
 * @brief ptmlDeviceGetGPIOStatus
 *
 * @param device device handles
 * @param GPIO status
 * @return ptmlReturn_t
 ********************************************/
ptmlReturn_t PTML_API ptmlDeviceGetGPIOStatus(ptmlDevice_t        device,
                                             unsigned int *status);
/**
 * @brief ptmlDeviceDumpCM3Regs
 *
 * @param device device handles
 * @param addr is CM3 reg addr
 * @param len is dump regs' length
 * @return ptmlReturn_t
 ********************************************/
ptmlReturn_t PTML_API ptmlDeviceDumpCM3Regs(ptmlDevice_t        device,
                                                   unsigned int addr,
                                                   unsigned int len,
                                                   unsigned int *value);
/**
 * @brief ptmlDeviceSetDumpTempSwitch
 *
 * @param device device handles
 * @param switch
 * @return ptmlReturn_t
 ********************************************/
ptmlReturn_t PTML_API ptmlDeviceSetDumpTempSwitch(ptmlDevice_t  device,
                                             unsigned int switchFlag);

int PTML_API ptmlDeviceGetBarInfo(ptmlDevice_t        device,
                                             unsigned int *size);

/**
 * @brief ptmlDeviceGetCPLDVersion
 *
 * @param device device handles
 * @param large version
 * @param small version
 * @return ptmlReturn_t
 ********************************************/
ptmlReturn_t PTML_API ptmlDeviceGetCPLDVersion(ptmlDevice_t        device,
                                              int *lver, int *sver);

ptmlReturn_t PTML_API ptmlDeviceGetProcessInfo(ptmlDevice_t        device,
                                              int processNum, struct ptProcessInfo *procInfo);
ptmlReturn_t PTML_API ptmlDeviceGetMaxPower(ptmlDevice_t  device,
                                           int *maxPower);
ptmlReturn_t PTML_API ptmlDeviceGetException(ptmlDevice_t  device,
                                           unsigned int *num);
/**
 * @brief ptmlPtlinkEnableAll
 *
 * @return ptmlReturn_t
 */
ptmlReturn_t PTML_API ptmlPtlinkEnableAll(void);

/**
 * @brief ptmlPtlinkDisableAll
 *
 * @return ptmlReturn_t
 */
ptmlReturn_t PTML_API ptmlPtlinkDisableAll(void);

/**
 * @brief ptmlPtlinkPortControl
 *
 * @device device id
 * @port port number
 * @ ops operation: en, disable ...
 * @return ptmlReturn_t
 */
ptmlReturn_t PTML_API ptmlPtlinkPortControl(ptmlDevice_t device,
                                            uint32_t     port,
                                            uint32_t     ops);

/**
 * @brief ptmlPtlinkPhytopoDetect
 *
 * @device device id
 * @size memory size
 * @buffer user buffer
 * @return ptmlReturn_t
 */
ptmlReturn_t PTML_API ptmlPtlinkPhytopoDetect(ptmlDevice_t device,
                                              uint32_t     size,
                                              void *       buffer);

/**
 * @brief ptmlEngineCollAssign
 *
 * @device device id
 * @coll_type collective type
 * @buffer user buffer
 * @size memory size
 * @return ptmlReturn_t
 */
ptmlReturn_t PTML_API ptmlEngineCollAssign(ptmlDevice_t device,
                                           uint32_t     coll_type,
                                           void *       buffer,
                                           uint32_t     size);
/**
 * @brief ptmlPtlinkGetConnectRelation
 *
 * @device1 device id
 * @device2 device id
 * @status the relationship between devices
 * @return ptmlReturn_t
 * @note  before use this api please use ptmlPtlinkEnableAll
 */
ptmlReturn_t PTML_API ptmlPtlinkGetConnectRelation(ptmlDevice_t device1,
                                     ptmlDevice_t device2,
                                     int *status) ;

/**
 * @brief ptmlGetPtlinkStatus
 *
 * @device device id
 * @port   port id
 * @status the status of the port
 * @return ptmlReturn_t
 * @note  before use this api please use ptmlPtlinkEnableAll
 */
ptmlReturn_t PTML_API ptmlGetPtlinkStatus(ptmlDevice_t device,
                                     int port,
                                     int *status);

/**
 * @brief ptmlGetPtlinkRemoteDevicePciInfo
 *
 * @device device id
 * @port   port id
 * @pciInfo pciInfo of remote device
 * @return ptmlReturn_t
 * @note  before use this api please use ptmlPtlinkEnableAll
 */
ptmlReturn_t PTML_API ptmlGetPtlinkRemoteDevicePciInfo(ptmlDevice_t device,
                                     int port,
                                     ptPciInfo_t *pciInfo);

/**
 * @brief ptmlGetErrorCodeToDescription
 *
 * @errCode errCode
 * @return errorDescription
 */
PTML_API const char *ptmlGetErrorCodeToDescription(int errorCode);

typedef enum ptmlEventType_enum {
  PTML_EVENT_TYPE_PSTATE,
  PTML_EVENT_TYPE_ALL,
} ptmlEventType_t;

typedef enum ptmlDeviceStateChange {
  PTMLDEVICE_GOOD_TO_BAD,
  PTMLDEVICE_BAD_TO_GOOD,
} ptmlDeviceStateChange_t;

typedef enum ptmlDeviceState {
  PTMLDEVICE_BAD,
  PTMLDEVICE_GOOD,
} ptmlDeviceState_t;

typedef enum ptmlEventStrategy {
  PTMLEVENT_UN_MONITOR,
  PTMLEVENT_MONITOR,
} ptmlEventStrategy_t;

typedef struct ptmlEventData {
  ptmlDevice_t  device;
  unsigned long eventType;
  unsigned long eventData;
} ptmlEventData_t;

typedef struct ptmlEvent {
  ptmlEventStrategy_t strategy;
  ptmlDeviceState_t   state;
} ptmlEvent_t;

typedef struct ptmlEventSet {
  ptmlEvent_t deviceEvent[TA_PT_NUM_MAX][PTML_EVENT_TYPE_ALL];
} ptmlEventSet_t;

ptmlReturn_t PTML_API ptmlEventSetCreate(ptmlEventSet_t **set);
ptmlReturn_t PTML_API ptmlEventSetFree(ptmlEventSet_t *set);
ptmlReturn_t PTML_API ptmlDeviceRegisterEvents(ptmlDevice_t    device,
                                      unsigned long   eventTypes,
                                      ptmlEventSet_t *set);

ptmlReturn_t PTML_API ptmlEventSetWait_v2(ptmlEventSet_t * set,
                                 ptmlEventData_t *data,
                                 unsigned int     timeoutms);

typedef int*  ptmlDeviceErrorCode;
PTML_API int ptmlGetDeviceLastError(ptmlDevice_t device,
      ptmlDeviceErrorCode DeviceErrorCode);

#ifdef __cplusplus
}
#endif  //! __cplusplus

#endif  //! _S2_ML_H_
