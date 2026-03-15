# cuda related
# export CUDA_HOME=/usr/local/cuda-10.0
# export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# path related
export PRJ_ROOT="${PWD}/../../.."
if [ -e "${PRJ_ROOT}/tools/venv/bin/activate" ]; then
    # shellcheck disable=SC1090
    . "${PRJ_ROOT}/tools/venv/bin/activate"
fi

MAIN_ROOT=$PWD/../../..
export PATH=$MAIN_ROOT/seq2seq_vc/bin:$PATH

# python related
export OMP_NUM_THREADS=1
export PYTHONIOENCODING=UTF-8
export MPL_BACKEND=Agg
