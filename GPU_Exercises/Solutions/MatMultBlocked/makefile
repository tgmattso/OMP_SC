#
#  USAGE:
#     make          ... to build the program
#     make test     ... to run the default test case
#
include ../../make.def

EXES= mm_testbed$(EXE)

MM_OBJS  = mm_testbed.$(OBJ) mm_utils.$(OBJ) mm_tst_cases.$(OBJ) \
           mm_gpu_block.$(OBJ) mm_gpu.$(OBJ) mm_cpu.$(OBJ)

all: $(EXES)

mm_testbed$(EXE): $(MM_OBJS) mm_utils.h
	$(CLINKER) $(CFLAGS) -o mm_testbed$(EXE) $(MM_OBJS) $(LIBS)

test: $(EXES)
	for i in $(EXES); do \
	   $(PRE)$$i; \
	done

clean:
	$(RM) $(EXES) *.$(OBJ) *.ptx *.cub

mm_utils.$(OBJ): mm_utils.h

.SUFFIXES:
.SUFFIXES: .c .cpp  .$(OBJ)

.c.$(OBJ):
	$(CC) $(CFLAGS) -c $<

.cpp.$(OBJ):
	$(CC) $(CFLAGS) -c $<
                                                       
