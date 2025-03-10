Currently writing bitstream is very slow due to the auto-regressive model. If you want to write bitstream, you need to build the arithmetic coder first.

# Build
* Build on Windows

    CMake and Visual Studio 2019 are needed.
    ```bash
    cd src
    mkdir build
    cd build
    conda activate $YOUR_PY38_ENV_NAME
    cmake ../cpp -G "Visual Studio 16 2019" -A x64
    cmake --build . --config Release
    ```

* Build on Linux (recommended)

    CMake and g++ are needed.
    ```bash
    sudo apt-get install cmake g++
    cd src
    mkdir build
    cd build
    conda activate $YOUR_PY38_ENV_NAME
    cmake ../cpp -DCMAKE_BUILD_TYPE=Release
    make -j
    ```
# Test
Please append this into your test command:
```
--write_stream True
```