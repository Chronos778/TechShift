from PIL import Image

# Load image and resize to 1x1 pixel
img = Image.open("kitty.jpg").resize((1, 1)).convert("RGB")

# Get pixel data (R, G, B)
r, g, b = img.getpixel((0, 0))

# Convert to RGB565 format
r5 = (r >> 3)  # 5 bits
g6 = (g >> 2)  # 6 bits
b5 = (b >> 3)  # 5 bits

rgb565 = (r5 << 11) | (g6 << 5) | b5  # 16-bit packed data

# Save as binary
with open("output.bin", "wb") as f:
    f.write(rgb565.to_bytes(115200, byteorder="big"))

print("Image converted to 2-byte RGB565 format!")
