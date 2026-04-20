import tensorrt as trt


logger = trt.Logger(trt.Logger.WARNING)

builder = trt.Builder(logger)

network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

parser = trt.OnnxParser(network,logger)

with open('OverNet.onnx','rb') as f:
    if not parser.parse(f.read()):
        for error in range(parser.num_errors):
            print(parser.get_error(error))


config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
config.set_flag(trt.BuilderFlag.FP16)



serialized_engine = builder.build_serialized_network(network,config)

with open('OverNet.trt','wb') as f:
    f.write(serialized_engine)
print('engine saved to OverNet.trt')