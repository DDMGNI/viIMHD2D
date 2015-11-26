
PYTHONPATH := $(CURDIR):${PYTHONPATH}
export PYTHONPATH

all:
	$(MAKE) -C imhd
	

clean:
	$(MAKE) clean -C imhd
