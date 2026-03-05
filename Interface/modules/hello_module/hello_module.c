#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/module.h>


static char *project = "default";
module_param(project, charp, 0000);
MODULE_PARM_DESC(project, "Name of the Overseer container project");

static int __init hello_init(void) {
    pr_info("[hello_module] Hello from Overseer project: %s\n", project);
    return 0;
}

static void __exit hello_exit(void) {
    pr_info("[hello_module] Goodbye from Overseer project: %s\n", project);
}

module_init(hello_init);
module_exit(hello_exit);


MODULE_LICENSE("GPL");
MODULE_AUTHOR("Kalzemic");
MODULE_DESCRIPTION("Basic hello world kernel module for Overseer.");