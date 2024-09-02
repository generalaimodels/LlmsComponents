Here are the mathematical notations for the techniques you've mentioned:

1. **Bitsandbytes (Quantization)**:
   $$
   \text{Quantized Weight} = \text{Round}\left(\frac{\text{Original Weight}}{\Delta}\right) \times \Delta
   $$
   where $\Delta$ is the quantization step size.

2. **GPT-Q (Quantized Transformer)**:
   $$
   \text{Quantized Output} = \text{Activation Function}\left(\text{Quantized Weight} \times \text{Quantized Input}\right)
   $$

3. **EETQ (Enhanced Efficient Quantization)**:
   $$
   \text{Quantized Weight} = \text{Clip}\left(\frac{\text{Original Weight} - \text{Bias}}{\Delta}\right) \times \Delta + \text{Bias}
   $$

4. **AWQ (Adaptive Weight Quantization)**:
   $$
   \text{Quantized Weight} = \text{Round}\left(\frac{\text{Original Weight} - \text{Min Weight}}{\text{Step Size}}\right) \times \text{Step Size} + \text{Min Weight}
   $$

5. **Marlin (Mixed Precision Quantization)**:
   $$
   \text{Mixed Precision Output} = \text{Activation Function}\left(\text{Quantized Weight}_{\text{low precision}} \times \text{Quantized Input}_{\text{high precision}}\right)
   $$

6. **FP8 (8-bit Floating Point)**:
   $$
   \text{FP8 Representation} = \text{Sign} \times 2^{\text{Exponent}} \times \text{Mantissa}
   $$

7. **Safetensors Weight Loading**:
   $$
   \text{Loaded Weight} = \text{Safe Load Function}\left(\text{Serialized Weight Data}\right)
   $$


   Here are the specific equations for 4-bit and 8-bit quantization using the `bitsandbytes` API from the Hugging Face Transformers module, incorporating the parameters provided:

### 8-bit Quantization

For 8-bit quantization in `bitsandbytes`, the process is typically as follows:

1. **Determine Quantization Range**:
   $$
   \text{Range}_{8\text{bit}} = 2^{8} - 1 = 255
   $$

2. **Quantize Weights**:
   $$
   \text{Quantized Weight}_{8\text{bit}} = \text{Round}\left(\frac{\text{Original Weight}}{\Delta_{8\text{bit}}}\right) \times \Delta_{8\text{bit}}
   $$
   where
   $$
   \Delta_{8\text{bit}} = \frac{\text{Max}_{\text{Original Weight}} - \text{Min}_{\text{Original Weight}}}{255}
   $$

3. **Post-Quantization Adjustments**:
   $$
   \text{Quantized Weight}_{8\text{bit}} = \text{Clip}\left(\text{Quantized Weight}_{8\text{bit}}, \text{Min}_{\text{Original Weight}}, \text{Max}_{\text{Original Weight}}\right)
   $$

### 4-bit Quantization

For 4-bit quantization in `bitsandbytes`, the process is typically as follows:

1. **Determine Quantization Range**:
   $$
   \text{Range}_{4\text{bit}} = 2^{4} - 1 = 15
   $$

2. **Quantize Weights**:
   $$
   \text{Quantized Weight}_{4\text{bit}} = \text{Round}\left(\frac{\text{Original Weight} - \text{Min}_{\text{Original Weight}}}{\Delta_{4\text{bit}}}\right) \times \Delta_{4\text{bit}} + \text{Min}_{\text{Original Weight}}
   $$
   where
   $$
   \Delta_{4\text{bit}} = \frac{\text{Max}_{\text{Original Weight}} - \text{Min}_{\text{Original Weight}}}{15}
   $$

3. **Post-Quantization Adjustments**:
   $$
   \text{Quantized Weight}_{4\text{bit}} = \text{Clip}\left(\text{Quantized Weight}_{4\text{bit}}, \text{Min}_{\text{Original Weight}}, \text{Max}_{\text{Original Weight}}\right)
   $$

### Specific API Parameters

- `load_in_8bit`: If `True`, enables 8-bit quantization.
- `load_in_4bit`: If `True`, enables 4-bit quantization.
- `bnb_4bit_compute_dtype`: Specifies the dtype used for computations with 4-bit quantized weights.
- `bnb_4bit_quant_type`: Defines the type of quantization (e.g., "fp4").
- `bnb_4bit_use_double_quant`: If `True`, uses double quantization to enhance precision.

### Example Using API Parameters

For 8-bit quantization:
$$
\text{Quantized Weight}_{8\text{bit}} = \text{Round}\left(\frac{\text{Original Weight}}{\frac{\text{Max} - \text{Min}}{255}}\right) \times \frac{\text{Max} - \text{Min}}{255}
$$

For 4-bit quantization:
$$
\text{Quantized Weight}_{4\text{bit}} = \text{Round}\left(\frac{\text{Original Weight} - \text{Min}}{\frac{\text{Max} - \text{Min}}{15}}\right) \times \frac{\text{Max} - \text{Min}}{15} + \text{Min}
$$



In the context of quantization with `bitsandbytes`, the ranges for 4-bit and 8-bit quantization are as follows:

### 8-bit Quantization

- **Unsigned 8-bit Integer (UInt8)**:
  $$
  \text{Range}_{8\text{bit}} = [0, 255]
  $$

- **Signed 8-bit Integer (Int8)**:
  $$
  \text{Range}_{8\text{bit}} = [-128, 127]
  $$

### 4-bit Quantization

- **Unsigned 4-bit Integer (UInt4)**:
  $$
  \text{Range}_{4\text{bit}} = [0, 15]
  $$

- **Signed 4-bit Integer (Int4)**:
  $$
  \text{Range}_{4\text{bit}} = [-8, 7]
  $$

### Summary

- **8-bit Unsigned Range**: [0, 255]
- **8-bit Signed Range**: [-128, 127]
- **4-bit Unsigned Range**: [0, 15]
- **4-bit Signed Range**: [-8, 7]



### Mathematical Equations

1. **Throughput**:
   $$
   \text{Throughput} = \frac{\text{Number of Operations}}{\text{Time}}
   $$
   where throughput is measured in operations per second (e.g., tokens per second).

2. **Latency**:
   $$
   \text{Latency} = \text{Time of Response} - \text{Time of Request}
   $$
   where latency is measured in milliseconds (ms).

3. **Bandwidth**:
   $$
   \text{Bandwidth} = \frac{\text{Total Data Transferred}}{\text{Time}}
   $$
   where bandwidth is measured in gigabytes per second (GB/s) or terabytes per second (TB/s).

4. **Memory Requirement**:
   $$
   M = \frac{P \times B}{Q}
   $$
   where
   - $ P$ = Model Parameters
   - $ B$ = Bytes per Parameter
   - $ Q$ = Quantization Factor