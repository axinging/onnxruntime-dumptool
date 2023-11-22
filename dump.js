const onnx = ort.OnnxProto.onnx;

// From 'https://cdn.jsdelivr.net/npm/long@5.2.3/index.js';
function isLong(obj) {
  return (obj && obj['__isLong__']) === true;
}

function sleep(ms) {
  let start = new Date().getTime();
  let expire = start + ms;
  while (new Date().getTime() < expire) {
  }
  return;
}

function writeObjectToFile(jsonObject, name, time = 200) {
  let object = jsonObject;
  const fileName = name;
  const a = document.createElement('a');
  if (object instanceof Map) {
    object = Object.fromEntries(object);
  }
  try {
    let jsonStr = name.split('.').pop() === 'jsonc' ? JSON.stringify([object]) :
                                                      JSON.stringify(object);
    const file = new Blob([jsonStr], {type: 'application/json'});
    a.href = URL.createObjectURL(file);
    a.download = fileName;
    a.click();
  } catch (e) {
    console.warn(" " + name + ", " + typeof(object));
    let jsonStr = "[" + Object.entries(object).map(el => JSON.stringify(el)).join(",") + "]";
    const file = new Blob([jsonStr], {type: 'application/json'});
    a.href = URL.createObjectURL(file);
    a.download = fileName;
    a.click();
    // throw e;
  }
  sleep(time);
}

function writeMapToFile(jsonObject, name, time = 200) {
  let object = jsonObject;
  const fileName = name;
  const a = document.createElement('a');
  if (object instanceof Map) {
    for (let [key, value] of object) {
      // console.log(key + ' = ' + value);
      writeObjectToFile(value, key + '.json');
    }
  }
}

function writeTypedArrayToFile(tyepdArray, name, time = 200) {
  let object = tyepdArray;
  const fileName = name;
  const a = document.createElement('a');
  if (object instanceof Map) {
    object = Object.fromEntries(object);
  }
  const file = new Blob([object], {type: 'application/json'});
  a.href = URL.createObjectURL(file);
  a.download = fileName;
  a.click();
  sleep(time);
}

async function readTypedArrayFromFile(fileUrl) {
  let response = await fetch(fileUrl);
  const blob = await response.blob();
  const tyepdArray = new Uint8Array(await blob.arrayBuffer());
  return tyepdArray;
}

async function readFromJsonFile(fileUrl) {
  const response = await fetch(fileUrl);
  const json = await response.json();
  if (json.type === 'float') {
    json.type = 'float32';
  }
  return json;
}

BigInt.prototype.toJSON = function() {
  return Number(this.toString());
};

function getParam(name) {
  name = name.replace(/[\[\]]/g, '\\$&');
  let regex = new RegExp('[?&]' + name + '(=([^&#]*)|&|#|$)', 'i');
  let results = regex.exec(window.location.href);
  if (!results) return null;
  if (!results[2]) return '';
  return decodeURIComponent(results[2].replace(/\+/g, ' '));
}

async function readObjectFromFile(fileUrl) {
  const response = await fetch(fileUrl);
  const blob = await response.blob();
  const blobObject = JSON.parse(await blob.text());
  return blobObject;
}

async function createOnnxModel(test) {
  const opsetImport = onnx.OperatorSetIdProto.create(test.opset);

  const operator = test.operator;
  const attribute = (test.attributes || []).map(attr => {
    const protoAttr = onnx.AttributeProto.create({name: attr.name});
    switch (attr.type) {
      case 'float':
        protoAttr.type = onnx.AttributeProto.AttributeType.FLOAT;
        protoAttr.f = attr.data;
        break;
      case 'int':
        protoAttr.type = onnx.AttributeProto.AttributeType.INT;
        protoAttr.i = attr.data;
        break;
      case 'string':
        protoAttr.type = onnx.AttributeProto.AttributeType.STRING;
        protoAttr.s = new TextEncoder().encode(attr.data);
        break;
      case 'floats':
        protoAttr.type = onnx.AttributeProto.AttributeType.FLOATS;
        protoAttr.floats = attr.data;
        break;
      case 'ints':
        protoAttr.type = onnx.AttributeProto.AttributeType.INTS;
        protoAttr.ints = attr.data;
        break;
      case 'strings':
        protoAttr.type = onnx.AttributeProto.AttributeType.STRINGS;
        protoAttr.strings = (attr.data).map(s => new TextEncoder().encode(s));
        break;
      default:
        throw new Error(`Unsupported attribute type: ${attr.type}`);
    }
    return protoAttr;
  });

  if (test.cases.length === 0) {
    throw new Error(
        `No test cases found for test: ${test.name} [${test.operator}]`);
  }
  const inputCount = test.cases[0].inputs.length;
  const outputCount = test.cases[0].outputs.length;
  if (test.cases.some(
          testCase => testCase.inputs.length !== inputCount ||
              testCase.outputs.length !== outputCount)) {
    throw new Error(`Test cases for test: ${test.name} [${
        test.operator}] must have the same number of inputs and outputs`);
  }

  const model = onnx.ModelProto.create();
  model.irVersion = onnx.Version.IR_VERSION;
  console.log(model.opsetImport);
  model.opsetImport.push(opsetImport);
  model.graph = onnx.GraphProto.create();

  model.graph.node = [onnx.NodeProto.create({
    input: test.cases[0].inputs.map((_, i) => `input_${i}`),
    output: test.cases[0].outputs.map((_, i) => `output_${i}`),
    opType: operator,
    domain: test.opset?.domain,
    name: operator,
    attribute
  })];

  // normalize input shape definitions
  let normalizedInputShapeDefinitions;
  if (!test.inputShapeDefinitions || test.inputShapeDefinitions === 'none') {
    // if inputShapeDefinitions is not specified, use undefined for all inputs
    normalizedInputShapeDefinitions = new Array(inputCount).fill(undefined);
  } else if (test.inputShapeDefinitions === 'rankOnly') {
    // check if all test cases have data
    if (test.cases.some(
            testCase =>
                testCase.inputs.some(input => !input.data || !input.dims))) {
      throw new Error(`Test cases for test: ${test.name} [${
          test.operator}] must have data for each inputs when inputShapeDefinitions is 'rankOnly'`);
    }

    // if inputShapeDefinitions is 'rankOnly', use semantic names for all
    // inputs. This means only rank is specified.
    normalizedInputShapeDefinitions = test.cases[0].inputs.map(
        (input, i) => input.dims.map((_, j) => `_input_${i}_d${j}`));

    // check if all test cases have the same rank for each inputs
    if (test.cases.some(
            testCase => testCase.inputs.some(
                (input, i) => input.dims.length !==
                    (test.cases[0].inputs[i]).dims.length))) {
      throw new Error(`Test cases for test: ${test.name} [${
          test.operator}] must have the same rank for each inputs in different test cases`);
    }
  } else if (test.inputShapeDefinitions === 'static') {
    // check if all test cases have data
    if (test.cases.some(
            testCase =>
                testCase.inputs.some(input => !input.data || !input.dims))) {
      throw new Error(`Test cases for test: ${test.name} [${
          test.operator}] must have data for each inputs when inputShapeDefinitions is 'rankOnly'`);
    }

    // if inputShapeDefinitions is 'static', use the shape of the first test
    // case for all inputs.
    normalizedInputShapeDefinitions =
        test.cases[0].inputs.map((input) => input.dims);

    // check if all test cases have the same shape for each inputs
    if (test.cases.some(
            testCase => testCase.inputs.some(
                (input, i) => TensorResultValidator.integerEqual(
                    input.dims, (test.cases[0].inputs[i]).dims)))) {
      throw new Error(`Test cases for test: ${test.name} [${
          test.operator}] must have the same shape for each inputs in different test cases`);
    }
  } else {
    // if inputShapeDefinitions is specified as an array, use it as is.
    // check if inputShapeDefinitions has the same number of inputs as test
    // cases
    if (test.inputShapeDefinitions &&
        test.inputShapeDefinitions.length !== inputCount) {
      throw new Error(`Input shape definitions for test: ${test.name} [${
          test.operator}] must have the same number of inputs`);
    }
    normalizedInputShapeDefinitions = test.inputShapeDefinitions;
  }

  model.graph.input = test.cases[0].inputs.map((input, i) => {
    const shapeDefinition = normalizedInputShapeDefinitions[i];
    const shape = shapeDefinition ? onnx.TensorShapeProto.create({
      dim: shapeDefinition.map(
          dim => onnx.TensorShapeProto.Dimension.create(
              typeof dim === 'string' ? {dimParam: dim} : {dimValue: dim}))
    }) :
                                    undefined;
    return onnx.ValueInfoProto.create({
      name: `input_${i}`,
      type: onnx.TypeProto.create({
        tensorType: onnx.TypeProto.Tensor.create(
            {elemType: tensorDataTypeStringToEnum(input.type), shape}),
      }),
    });
  });

  model.graph.output =
      test.cases[0].outputs.map((output, i) => onnx.ValueInfoProto.create({
        name: `output_${i}`,
        type: onnx.TypeProto.create({
          tensorType: onnx.TypeProto.Tensor.create(
              {elemType: tensorDataTypeStringToEnum(output.type)}),
        }),
      }));

  model.graph.name = test.name;

  const backendHint = test.backend;
  const loadedData = onnx.ModelProto.encode(model).finish();

  const session = await ort.InferenceSession.create(
      loadedData, {executionProviders: [backendHint]});
  return session;
}

async function runOnnxProtoOp(graphPlan, session) {
  const testCase = graphPlan['cases'][0];
  const feeds = {};
  const fetches = [];
  testCase.inputs.forEach((input, i) => {
    if (input.data) {
      let data = input.data;
      if (input.type === 'uint64') {
        data = BigUint64Array.from(input.data.map(BigInt));
      } else if (input.type === 'int64') {
        data = BigInt64Array.from(input.data.map(BigInt));
      }
      feeds[`input_${i}`] = new ort.Tensor(input.type, data, input.dims);
    }
  });

  const outputs = [];
  const expectedOutputNames = [];
  testCase.outputs.forEach((output, i) => {
    if (output.data) {
      let data = output.data;
      if (output.type === 'uint64') {
        data = BigUint64Array.from(output.data.map(BigInt));
      } else if (output.type === 'int64') {
        data = BigInt64Array.from(output.data.map(BigInt));
      }
      outputs.push(new ort.Tensor(output.type, data, output.dims));
      expectedOutputNames.push(`output_${i}`);
      fetches.push(`output_${i}`);
    }
  });

  const results = await session.run(feeds, fetches);
  return results;
}

const tensorDataTypeStringToEnum = (type) => {
  switch (type) {
    case 'int8':
      return 3 /* DataType.int8 */;
    case 'uint8':
      return 2 /* DataType.uint8 */;
    case 'bool':
      return 9 /* DataType.bool */;
    case 'int16':
      return 5 /* DataType.int16 */;
    case 'uint16':
      return 4 /* DataType.uint16 */;
    case 'int32':
      return 6 /* DataType.int32 */;
    case 'uint32':
      return 12 /* DataType.uint32 */;
    case 'float16':
      return 10 /* DataType.float16 */;
    case 'float32':
      return 1 /* DataType.float */;
    case 'float64':
      return 11 /* DataType.double */;
    case 'string':
      return 8 /* DataType.string */;
    case 'int64':
      return 7 /* DataType.int64 */;
    case 'uint64':
      return 13 /* DataType.uint64 */;
    default:
      throw new Error(`unsupported data type: ${type}`);
  }
};

function tensorDimsFromProto(dims) {
  // get rid of Long type for dims
  return dims.map(d => isLong(d) ? d.toNumber() : d);
}

// opset is array: [{domain: '', version: 8}, ]
function getOpset(opType, opsets) {
  let opset = {'domain': 'com.microsoft', 'version': 1};
  if (opType === 'Add' || opType === 'Conv' || opType === 'Shape' ||
      opType === 'Reshape' || opType === 'Gather' || opType === 'Unsqueeze' ||
      opType === 'Concat' || opType === 'GlobalAveragePool' ||
      opType === 'Slice' || opType === 'Cast' || opType === 'Softmax' ||
      opType === 'MatMul' || opType === 'Sub' || opType === 'Mul' ||
      opType === 'Add' || opType === 'Div' || opType === 'LayerNormalization' ||
      opType === 'Transpose' || opType === 'Gemm' || opType === 'LeakyRelu' ||
      opType === 'MaxPool' || opType === 'BatchNormalization' ||
      opType === 'ReduceMean'||
      opType === 'Pow'||
      opType === 'Tanh'||
      opType === 'Sqrt') {
    opset.domain = '';
  }

  let versionFound = false;
  for (const item of opsets) {
    if (opset.domain === item.domain) {
      opset.version = item.version;
      versionFound = true;
    }
  }
  if (!versionFound) {
    throw new Error(
        'Not find domain: ' + JSON.stringify(opset.domain) + ' in ' +
        JSON.stringify(opsets));
  }
  return opset;
}

function tensorDataTypeFromProto(typeProto) {
  switch (typeProto) {
    case onnx.TensorProto.DataType.INT8:
      return 'int8';
    case onnx.TensorProto.DataType.UINT8:
      return 'uint8';
    case onnx.TensorProto.DataType.BOOL:
      return 'bool';
    case onnx.TensorProto.DataType.INT16:
      return 'int16';
    case onnx.TensorProto.DataType.UINT16:
      return 'uint16';
    case onnx.TensorProto.DataType.INT32:
      return 'int32';
    case onnx.TensorProto.DataType.UINT32:
      return 'uint32';
    case onnx.TensorProto.DataType.FLOAT:
      return 'float32';
    case onnx.TensorProto.DataType.DOUBLE:
      return 'float64';
    case onnx.TensorProto.DataType.STRING:
      return 'string';

    // For INT64/UINT64, reduce their value to 32-bits.
    // Should throw exception when overflow
    case onnx.TensorProto.DataType.INT64:
      return 'int64';
    case onnx.TensorProto.DataType.UINT64:
      return 'uint64';

    default:
      throw new Error(
          `unsupported data type: ${onnx.TensorProto.DataType[typeProto]}`);
  }
}

async function runGraphPlan(graphPlan) {
  // ort.env.debug = true
  // ort.env.logLevel = 'verbose';

  const session = await createOnnxModel(graphPlan);
  const result = await runOnnxProtoOp(graphPlan, session);
  return result;
}

async function loadModel(arg) {
  const model = new ort.Model();
  if (typeof arg === 'string') {
    const isOrtFormat = arg.endsWith('.ort');
    if (typeof process !== 'undefined' && process.versions &&
        process.versions.node) {
      // node
      const buf = await readFile(arg);
      model.load(buf);
    } else {
      // browser
      const response = await fetch(arg);
      const buf = await response.arrayBuffer();
      model.load(new Uint8Array(buf));
    }
  } else if (!ArrayBuffer.isView(arg)) {
    // load model from ArrayBuffer
    const arr = new Uint8Array(arg, byteOffset || 0, length || arg.byteLength);
    model.load(arr);
  } else {
    model.load(arg);
  }
  return model;
}

function convertArrayToBigInt64Array(array) {
  const bigint64array = new BigInt64Array(array.length);
  for (var i = 0; i < array.length; i++) {
    bigint64array[i] = BigInt(array[i]);
  }
  return bigint64array;
}

function compareIgnoreType(reference, result) {
  const isResultInt64 = result instanceof BigInt64Array;
  const referenceInt64 =
      isResultInt64 ? convertArrayToBigInt64Array(reference) : reference;
  if (isResultInt64) {
    return (
        JSON.stringify(referenceInt64.sort()) ===
        JSON.stringify(result.sort()));
  }
  return compare(referenceInt64, Array.from(result));
}

function getDirInfo(modelName, graphOptimizationLevel) {
  const optimizedModelName = modelName + '-' + graphOptimizationLevel + '.json';
  const optimizedModelDataName =
      modelName + '-' + graphOptimizationLevel + '-data.json';
  const modelDir = './ort-models/';
  const modelDataDir =
      modelDir + modelName + '-' + graphOptimizationLevel + '/';
  return [modelDir, modelDataDir, optimizedModelName, optimizedModelDataName];
}

export class OnnxDumpData {
  constructor(modelName, graphOptimizationLevel, dumpOrCmp) {
    // mode 0: will not create any  data file, output a result file. mode 1:
    // generate data file, to use these file in mode. mode 2: copy file
    // generated by mode 1 modelDataDir.
    this.dumpOrCmp = Number(dumpOrCmp);
    // useFile will speed up when you have to dump the same model rerpeatedly.
    this.useFile = dumpOrCmp != 0;
    // Store all weights, input or output data.
    this.dumpDataMap = new Map();
    // Uint8Array, can be decode by onnx proto directly.
    this.optimizedModelBuffer = null;
    // 'disabled'|'basic'|'extended'|'all'.
    this.graphOptimizationLevel = graphOptimizationLevel ?? 'all';
    // From https://github.com/webatintel/ort-toolkit/blob/main/models.js.
    this.modelName = modelName;
    // modelDir: directory of .onnx model file. When useFile, include weights
    // and i/o data. optimizedModelName: optimized, json format ort model file.
    // optimizedModelDataName:  weights and i/o data of model.
    const [modelDir, modelDataDir, optimizedModelName, optimizedModelDataName] =
        getDirInfo(modelName, graphOptimizationLevel);
    Object.assign(
        this,
        {modelDir, modelDataDir, optimizedModelName, optimizedModelDataName});
    this.modelUrl = this.modelDir + this.modelName + '.onnx';
    // Type ort.Model.
    this.model = null;
    this.referenceBackend = 'wasm';
    this.actualBackend = getParam('ep') || 'webgpu';
  }

  release() {
    // TODO.
  }

  async setupWeights() {
    const modelProto = onnx.ModelProto.decode(this.optimizedModelBuffer);
    for (const i of modelProto.graph.initializer) {
      const tensor = {
        'data': Array.from(ort.JsTensor.Tensor.fromProto(i).data),
        'dims': tensorDimsFromProto(i.dims),
        'type': tensorDataTypeFromProto(i.dataType),
      };
      const regName = i.name.replace(/\//g, '_').replace(/:/g, '_');
      // console.log('namedebug: ' + regName);
      // writeObjectToFile(tensor, regName);
      this.dumpDataMap.set(regName, tensor);
      ;
    }
  }

  // Get the input output data.
  async setupInputOutputs() {
    if (window.onnxDumpBlobUrlMap == null) {
      throw new Error('window.onnxDumpBlobUrlMap is NULL!');
    }
    const blobUrlMap = window.onnxDumpBlobUrlMap;
    for (const [key, value] of blobUrlMap.entries()) {
      // console.log('namedebug: ' + key);
      const blobObject = await readObjectFromFile(value);
      this.dumpDataMap.set(key, blobObject);
    }
  }

  async setup(onnxModelInferenceFn) {
    window.onnxDump = 2;
    const optimizedModelBuffer = await this.getOptimizedModel();
    const optimizedModelName = this.optimizedModelName;
    window.onnxDump = 0;
    if (this.useFile) {
      writeTypedArrayToFile(optimizedModelBuffer, optimizedModelName);
    }
    // Generate weights data.
    console.log('Dump - Generate weights data.');
    await this.setupWeights(optimizedModelBuffer);
    console.log('Dump - Generate input output data.');
    // Generate other dump data: input, output.
    window.onnxDump = 1;
    await onnxModelInferenceFn(
        'performance', this.referenceBackend, this.modelUrl,
        this.graphOptimizationLevel);
    window.onnxDump = 0;
    await this.setupInputOutputs();
  }

  async getOptimizedModel() {
    // const modelName = this.modelName;
    // const modelDataDir = this.modelDataDir;
    console.log('Dump - Optimize model begin.');
    const graphOptimizationLevel = this.graphOptimizationLevel;
    const optimizedModelFilePath =
        this.modelName + '-' + graphOptimizationLevel + '.onnx';
    let session;

    try {
      const option = {
        executionProviders: [
          {
            name: this.referenceBackend,
          },
        ],
        graphOptimizationLevel: graphOptimizationLevel,
        optimizedModelFilePath: optimizedModelFilePath,
      };
      session = await ort.InferenceSession.create(
          this.modelDir + this.modelName + '.onnx', option);
      console.log('Dump - Optimize model end.');

    } catch (e) {
      console.error(`Failed to inference ONNX model: ${e}.`);
    }

    console.log(window.onnxDumpOptmizedModelBlobUrl);
    const response = await fetch(window.onnxDumpOptmizedModelBlobUrl);
    const blob = await response.blob();
    this.optimizedModelBuffer = new Uint8Array(await blob.arrayBuffer());
    // await session.release();
    return this.optimizedModelBuffer;
  }

  save() {
    const optimizedModelDataName = this.optimizedModelDataName;
    const dumpDataMap = this.dumpDataMap;
    if (this.useFile) {
      // writeObjectToFile works on mobilenet, not on albert. For albert, file
      // is too big, need to save it by node.
      writeMapToFile(dumpDataMap, optimizedModelDataName);
    }
  }

  async restore() {
    if (this.useFile) {
      console.log(this.optimizedModelName);
      if (this.optimizedModelBuffer == null) {
        this.optimizedModelBuffer = await readTypedArrayFromFile(
            this.modelDataDir + this.optimizedModelName);
      }
      console.log(this.optimizedModelBuffer);
      // when cmp only, this means the dump data is from seperated file.
      this.dumpDataMap = this.dumpOrCmp == 2 ?
          null :
          await readObjectFromFile(
              this.modelDataDir + this.optimizedModelDataName);
    }
  }

  async compare() {
    this.model = await loadModel(this.optimizedModelBuffer);
    await this.compareModel();
  }

  async setupGraphPlan(node) {
    const dumpDataMap = this.dumpDataMap;
    const modelName = this.modelName;
    const model = this.model;

    const nodePlan = {name: node.name};
    nodePlan.inputs = [];
    nodePlan.outputs = [];
    const inputShapeDefinitions = [];
    console.log(
        modelName + ', dump data ismap: ' + (dumpDataMap instanceof Map));
    for (const inputName of node.inputNames) {
      const inputData = await this.getData(inputName, node);
      nodePlan.inputs.push(inputData);
    }

    for (const outputName of node.outputNames) {
      let outputData = await this.getData(outputName, node);
      nodePlan.outputs.push(outputData);
    }

    for (const input of nodePlan['inputs']) {
      inputShapeDefinitions.push((input['dims']));
    }
    const attributs = [];
    node.attributes._attributes.forEach((value, key) => {
      attributs.push({'name': key, 'data': value[0], 'type': value[1]});
    });

    const opset = getOpset(node.opType, model._opsets);
    console.log(
        node.opType + ', ' +
        'domain: ' + JSON.stringify(opset));
    const graphPlan = {
      'name': node.opType,
      'operator': node.opType,
      'attributes': attributs,
      'inputShapeDefinitions': inputShapeDefinitions,
      'cases': [
        nodePlan,
      ],
      'backend': this.actualBackend,
      'opset': opset,
    };

    return graphPlan;
  }

  async getData(inputName, node) {
    const dumpDataMap = this.dumpDataMap;
    let data;
    const isMap = dumpDataMap instanceof Map;
    const regName = inputName.replace(/\//g, '_').replace(/:/g, '_');
    try {
      data = await readFromJsonFile(this.modelDataDir + regName + '.json');
    } catch (err) {
      data = isMap ? dumpDataMap.get(regName) : dumpDataMap[regName];
    } finally {
      if (data == null) {
        console.error(
            ('Can not find input or output: ' + node.name + ', ' +
             JSON.stringify(node)));
        return null;
      }
      if (data.type === 'float') {
        data.type = 'float32';
      }
    }
    return data;
  }

  async compareSingleNode(node) {
    const graphPlan = await this.setupGraphPlan(node);
    if (graphPlan == null) {
      return;
    }
    const result1 = await runGraphPlan(graphPlan);
    let reference = graphPlan['cases'][0]['outputs'][0].data;
    const compareResult =
        compareIgnoreType(reference, result1.output_0.cpuData);

    const compareSummary = {
      result: compareResult,
      opType: graphPlan['name'],
      node: graphPlan['cases'][0]['name']
    };
    if (compareResult == false) {
      console.log('Compare reference : ' + JSON.stringify(reference));
      console.log(
          'Compare result : ' +
          JSON.stringify(Array.from(result1.output_0.cpuData)));
      console.error(
          'Wasm vs ' + graphPlan['backend'] + ', compare result=' +
          compareResult + ', failed node: ' + graphPlan['name'] + ', ' +
          graphPlan['cases'][0]['name'] + ', inputShapeDefinitions = ' +
          JSON.stringify(graphPlan['inputShapeDefinitions']));
      writeObjectToFile(
          graphPlan,
          graphPlan['cases'][0]['name'] + '-' + this.graphOptimizationLevel +
              '.jsonc');
    }
    return compareSummary;
  }

  async compareModel() {
    console.log('Compare - Begin.');
    const model = this.model;
    const dumpDataMap = this.dumpDataMap;
    const modelName = this.modelName;
    const nodes = model.graph._nodes;
    let testNode = getParam('node');
    if (testNode) {
      for (const node of nodes) {
        if (testNode && node.name === testNode) {
          await this.compareSingleNode(node);
          break;
        }
      }
    } else {
      const compareSummaries = [];
      for (const node of nodes) {
        const compareSummary =
            await this.compareSingleNode(node, dumpDataMap, modelName, model);
            if (compareSummary.result === false) {
              compareSummaries.push(compareSummary);
            }
      }
      writeObjectToFile(
          {
            model: this.modelName,
            backend: `${this.referenceBackend} vs ${this.actualBackend}`,
            summary: compareSummaries
          },
          modelName + '-summary.json');
    }
    console.log('Compare - End.');
  }
}

/*
 * This tool will compare onnx model node results of two backends, such as wasm
 * and webgpu. Then output the incorrect nodes as jsonc files, these files can
 * be used as ort unit test.
 *
 * Before run, create a folder like example\modeldata\tinyyolov2-8-disabled,
 * then copy the .onnx into it.
 *
 * Usage:
 * modelName: name from
 * https://github.com/webatintel/ort-toolkit/blob/main/models.js, such as
 * albert-base-v2. onnxModelInferenceFn: a function which run whole inference on
 * model specificed by modelName. graphOptimizationLevel:
 * 'disabled'|'basic'|'extended'|'all'. dumpOrCmp: mode 0: will not create any
 * data file, output a result file. mode 1: generate data file, to use these
 * file in mode. mode 2: copy file generated by mode 1 to
 * example\modeldata\tinyyolov2-8-disabled. You can run it by default or mode 1
 * + mode 2.
 *
 * Known bugs/todos:
 * mobilenetv2-12-disabled not work.
 * Add node support.
 */

export async function dump(
    modelName, onnxModelInferenceFn, graphOptimizationLevel = 'disabled',
    dumpOrCmp = 0) {
  const useFile = dumpOrCmp != 0;
  const dumpBeginTime = performance.now();
  const dumpDataMap =
      new OnnxDumpData(modelName, graphOptimizationLevel, dumpOrCmp);
  if (dumpOrCmp != 2) {
    await dumpDataMap.setup(onnxModelInferenceFn);
    if (useFile) {
      dumpDataMap.save();
    }
  }
  if (dumpOrCmp != 1) {
    if (useFile) {
      await dumpDataMap.restore();
    }
    await dumpDataMap.compare();
    dumpDataMap.release();
  }
  console.log(
      'Dump time: ' + Math.round((performance.now() - dumpBeginTime) / 1000) +
      's.');
}
