To install FFmpeg without using `sudo` and assuming you have already downloaded the source code, follow these steps:

### 1. Download and Extract FFmpeg

If you haven’t already, download and extract the FFmpeg source code:

```bash
wget https://ffmpeg.org/releases/ffmpeg-5.1.2.tar.gz
tar -xzf ffmpeg-5.1.2.tar.gz
cd ffmpeg-5.1.2
```

### 2. Configure FFmpeg

Before configuring, ensure you have installed any necessary dependencies (like `nasm`). If you're building from source, you'll need `nasm` or `yasm`.

Run the configure script with the desired prefix (the installation directory):

```bash
./configure --prefix=/scratch/hemanth/Hemanth/ffmpeg
```

If you encounter issues with dependencies, ensure they are properly installed and located.

### 3. Compile FFmpeg

Compile the FFmpeg source code:

```bash
make
```

### 4. Install FFmpeg

Install FFmpeg to the specified prefix directory:

```bash
make install
```

### 5. Update PATH

Add the FFmpeg binary directory to your `PATH`:

```bash
export PATH=/scratch/hemanth/Hemanth/ffmpeg/bin:$PATH
```

### 6. Verify FFmpeg Installation

Check if FFmpeg is installed correctly:

```bash
ffmpeg -version
```

### Summary

1. **Download and Extract**: Use `wget` and `tar`.
2. **Configure**: Run `./configure` with the `--prefix` option.
3. **Compile**: Use `make`.
4. **Install**: Run `make install`.
5. **Update PATH**: Add the binary directory to `PATH`.
6. **Verify**: Check the version with `ffmpeg -version`.

If you follow these steps and still encounter issues, please provide specific error messages or problems you’re facing.