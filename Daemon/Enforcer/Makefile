obj-m += overseer_enforcer.o 

overseer_enforcer-objs := enforcer.o server.o process_worker.o

all:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules

clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean