CONFIG_DIR = /etc/overseer
BINARY_NAME = overseer
INSTALL_DIR = /usr/local/bin
MODULE_SRC_DIR = modules
build:
	go build -o $(BINARY_NAME) overseer.go

install: build init
	sudo mv $(BINARY_NAME) $(INSTALL_DIR)/$(BINARY_NAME)

init:
	sudo mkdir -p $(CONFIG_DIR)/projects
	sudo mkdir -p $(CONFIG_DIR)/modules
	@echo "🧠 Building kernel modules from local ./$(MODULE_SRC_DIR)/"
	@for dir in $(MODULE_SRC_DIR)/*/; do \
		if [ -f "$$dir/Makefile" ]; then \
			echo "📦 Building module in $$dir..."; \
			$(MAKE) -C "$$dir"; \
			sudo cp "$$dir"/*.ko $(CONFIG_DIR)/modules/; \
		fi \
	done
	
clean:
	@echo "🧹 Cleaning up..."
	rm -f $(BINARY_NAME)
	sudo rm -rf $(CONFIG_DIR)
	@echo "🧼 Cleaning up kernel module builds..."
	@for dir in $(MODULE_SRC_DIR)/*/; do \
		if [ -f "$$dir/Makefile" ]; then \
			echo "🧽 Cleaning $$dir..."; \
			$(MAKE) -C "$$dir" clean; \
		fi \
	done
	
uninstall:
	sudo rm -f $(INSTALL_DIR)/$(BINARY_NAME)
