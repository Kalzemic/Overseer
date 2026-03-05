Here's a setup manual for Overseer. 




Overseer Board Driver:

Connect the Overseer processor board to the pc via a pcie port. 


the board should come with a device driver installed under /dev/overseer. 



This device driver is required to pass information from the pc to the board and back. 

to enable the host driver (pc end) you must execute 

make persist 

at the .ko file location. this should have a /dev/overseer file appear on the pc. 


once the driver is set up, memory can be shared from the pc to the Overseer board and back. 

