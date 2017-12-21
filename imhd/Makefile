

all:
	python setup.py build_ext --inplace
	
	$(MAKE) -C integrators
	

clean:
	$(MAKE) clean -C integrators

	rm -f *.c
	rm -f *.so
	rm -fR build
