# Configs

## Install tensorflow without root

```python
lspci | grep -i nvidia # check the version of GPU

03:00.0 3D controller: NVIDIA Corporation GK110BGL [Tesla K40c] (rev a1)
04:00.0 3D controller: NVIDIA Corporation GK110BGL [Tesla K40c] (rev a1)
84:00.0 VGA compatible controller: NVIDIA Corporation GM107GL [Quadro K620] (rev a2)
```

## Install cmake 
download the latest cmake
move to the position folder
```shell
./configure
make
make install
```
Note that you need to change the path to set it as higher priority if there are more than one "cmake" 

## Change the path temperally
```shell
export PATH="path1:path2:path3"
echo $PATH # check your path
export PATH=/home/yunqiuxu/anaconda2/bin:$PATH
export PATH=/home/yunqiuxu/cmake/usr/local/bin:$PATH
export PATH=/home/yunqiuxu/bazel_local/output:$PATH
```
## Install tensorflow / cv2 / keras /caffe
```shell
conda install -c anaconda tensorflow-gpu=1.1.0
conda install -c menpo opencv
conda install keras
conda install -c conda-forge caffe=1.0.0rc5
```

