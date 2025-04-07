import argparse

def convert_to_header(input_file, output_file):
    with open(input_file, "rb") as f:
        data = f.read()

    header_lines = []
    header_lines.append("// Auto-generated header file from " + input_file)
    header_lines.append("unsigned char model_tflite[] = {")
    
    # Process data in chunks (e.g., 12 bytes per line)
    bytes_per_line = 12
    for i in range(0, len(data), bytes_per_line):
        chunk = data[i:i+bytes_per_line]
        # Format each byte as 0x..
        hex_values = ", ".join(f"0x{b:02x}" for b in chunk)
        header_lines.append("    " + hex_values + ",")
    
    header_lines.append("};")
    header_lines.append(f"unsigned int model_tflite_len = {len(data)};")
    
    with open(output_file, "w") as f:
        f.write("\n".join(header_lines))
    
    print(f"Header file saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a .lite (TensorFlow Lite) model file to a C header file for embedding in microcontroller projects.")
    parser.add_argument("input_file", help="Path to the .lite file (e.g., model.tflite)")
    parser.add_argument("output_file", help="Output header file path (e.g., model_data.h)")
    args = parser.parse_args()
    
    convert_to_header(args.input_file, args.output_file)
