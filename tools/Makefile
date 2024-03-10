PYTHON:= python3
CUDA_VERSION:= 11.7
PYTORCH_VERSION:= 1.13.1
DOT:= .
.PHONY: all clean show_variables seq2seq_vc s3prl_vc pytorch uninstall_pytorch monotonic_align

all: show_variables virtualenv.done pytorch.done seq2seq_vc.done s3prl_vc.done

seq2seq_vc: seq2seq_vc.done

s3prl_vc: s3prl_vc.done

pytorch: pytorch.done

show_variables:
	@echo PYTHON=$(PYTHON)
	@echo CUDA_VERSION=$(CUDA_VERSION)
	@echo PYTORCH_VERSION=$(PYTORCH_VERSION)

virtualenv.done: show_variables
	test -d venv || $(PYTHON) -m venv venv
	. venv/bin/activate; cd ../; pip install -U pip
	# install numpy here since python3.6 is not supported in > 1.20
	. venv/bin/activate; cd ../; pip install numpy
	touch virtualenv.done

pytorch.done: virtualenv.done
ifeq ($(CUDA_VERSION),)
	. venv/bin/activate; pip install torch==$(PYTORCH_VERSION) \
		-f https://download.pytorch.org/whl/cpu/stable.html
else
	. venv/bin/activate; pip install torch==$(PYTORCH_VERSION) \
		-f https://download.pytorch.org/whl/cu$(subst $(DOT),,$(CUDA_VERSION))/torch_stable.html
endif
	touch pytorch.done

seq2seq_vc.done: virtualenv.done pytorch.done
	. venv/bin/activate; cd ../; pip install -e .
	touch seq2seq_vc.done

s3prl_vc.done: seq2seq_vc.done
	. venv/bin/activate; cd ../; pip install -U s3prl-vc
	touch s3prl_vc.done

apex.done: virtualenv.done pytorch.done
	git clone https://github.com/NVIDIA/apex.git
	. venv/bin/activate; cd apex; \
		pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
	touch apex.done

monotonic_align:
	. venv/bin/activate; cd ../seq2seq_vc/monotonic_align; python setup.py build_ext --inplace; cd ../

clean:
	rm -fr venv apex *.done
	find -iname "*.pyc" -delete

uninstall_pytorch:
	. venv/bin/activate; cd ../; pip uninstall torch
