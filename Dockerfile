FROM gcc
RUN apt-get update && apt-get install -y libc6-dbg gdb valgrind cmake
RUN wget http://fftw.org/fftw-3.3.10.tar.gz && tar -xzf fftw-3.3.10.tar.gz \
    && cd fftw-3.3.10 && ./configure && make && make install
RUN wget https://github.com/google/benchmark/archive/refs/tags/v1.6.1.zip && unzip v1.6.1.zip \
    && cd benchmark-1.6.1 && cmake -E make_directory "build" \
    && cmake -E chdir "build" cmake -DBENCHMARK_DOWNLOAD_DEPENDENCIES=on -DCMAKE_BUILD_TYPE=Release ../ \
    && cmake --build "build" --config Release --target install
WORKDIR /work
COPY entrypoint.sh entrypoint.sh
ENTRYPOINT [ "./entrypoint.sh" ]
