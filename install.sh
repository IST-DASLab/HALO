set -e

# install the requirements and the main repo
pip install -r requirements.txt

pip install wheel
# make sure torch is using the latest version
pip install -U torch torchvision
# install flash-attn with no build isolation
pip install flash-attn --no-build-isolation

# install this repository
pip install -e .

# install int8 kernels
cd gemm-int8
pip install -e .
cd ../

# install fp8 kernels
cd gemm-fp8
pip install -e .
cd ../

# (optional) install baseline kernels (JetFire)
git clone https://github.com/thu-ml/Jetfire-INT8Training.git
cd Jetfire-INT8Training/JetfireGEMMKernel
python setup.py install
cd ../..

# # (optional) install HALO peft
cd peft && pip install -e . && cd ..
