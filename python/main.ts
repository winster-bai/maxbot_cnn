//% color="#f5ae43" iconWidth=50 iconHeight=40
namespace cnn{

    //% block="初始化tensorflow模块" blockType="command" 
    export function feat_init(parameter: any, block: any){

        Generator.addImport("import tensorflow as tf");
        Generator.addImport("from tensorflow import keras");


    }

    //% block="获取目录[OBJ]下的所有文件和子文件夹名称" blockType="reporter"
    //% OBJ.shadow="normal" OBJ.defl="path_name"
    export function listdir(parameter: any, block: any){
        let obj=parameter.OBJ.code;  
        Generator.addCode(`tf.io.gfile.listdir(${obj})`)
    }

    //% block="对象名[OBJ] 读取文件[FILE]" blockType="command"
    //% OBJ.shadow="normal" OBJ.defl="obj_name"
    //% FILE.shadow="normal" FILE.defl="file_name"
    export function read_file(parameter: any, block: any){
        let obj=parameter.OBJ.code;  
        let file=parameter.FILE.code;  
        Generator.addCode(`${obj} = tf.io.read_file(str(${file}))`)
    }

    //% block="加载图片数据训练集[TRAIN]和验证集[VAL]数据 数据源[PATH] 比例[SPLIT] 输入大小[INPUT]" blockType="command"
    //% TRAIN.shadow="normal"   TRAIN.defl="train_generator"
    //% VAL.shadow="normal"   VAL.defl="val_generator"
    //% PATH.shadow="normal"   PATH.defl="path_name"
    //% SPLIT.shadow="normal"   SPLIT.defl="0.2"
    //% INPUT.shadow="dropdown"   INPUT.options="SIZE"
    export function dataset_load(parameter: any, block: any){
        let train = parameter.TRAIN.code
        let val = parameter.VAL.code
        let path = parameter.PATH.code
        let split = parameter.SPLIT.code
        let input = parameter.INPUT.code
        Generator.addImport("from tensorflow.keras.preprocessing.image import ImageDataGenerator");
        Generator.addCode(`
datagen = ImageDataGenerator(rescale=1./255, validation_split=${split})

# 加载训练集数据
${train} = datagen.flow_from_directory(
    ${path},
    target_size=(${input}),
    color_mode="grayscale",
    batch_size=32,
    class_mode="categorical",
    subset="training"
)

# 加载验证集数据
${val} = datagen.flow_from_directory(
    ${path},
    target_size=(${input}),
    color_mode="grayscale",
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)
        `)
    }


    //% block="加载音频数据训练集[TRAIN]和验证集[VAL]数据 数据源[PATH] 比例[SPLIT] 输出序列长度[INPUT]" blockType="command"
    //% TRAIN.shadow="normal"   TRAIN.defl="train_generator"
    //% VAL.shadow="normal"   VAL.defl="val_generator"
    //% PATH.shadow="normal"   PATH.defl="path_name"
    //% SPLIT.shadow="normal"   SPLIT.defl="0.2"
    //% INPUT.shadow="normal"   INPUT.defl="16000"
    export function dataset_load_audio(parameter: any, block: any){
        let train = parameter.TRAIN.code
        let val = parameter.VAL.code
        let path = parameter.PATH.code
        let split = parameter.SPLIT.code
        let input = parameter.INPUT.code
        Generator.addCode(`
${train}, ${val} = tf.keras.utils.audio_dataset_from_directory(
    directory=${path},
    batch_size=32,
    validation_split=${split},
    seed=0,
    output_sequence_length=${input},
    subset='both')
        `)
    }


    //% block="创建顺序模型[OBJ]" blockType="reporter"
    //% OBJ.shadow="normal" OBJ.defl=""
    export function model_init_by_file(parameter: any, block: any){
        let obj=parameter.OBJ.code;  
        Generator.addCode(`keras.Sequential(name="${obj}")`)
    }

    //% block="对象名[MOD2]导入模型[MOD1]" blockType="command"
    //% MOD1.shadow="normal"   MOD1.defl="modelname"
    //% MOD2.shadow="normal"   MOD2.defl="model"
    export function mod_load(parameter: any, block: any){
        let mod1 = parameter.MOD1.code
        let mod2 = parameter.MOD2.code
        Generator.addCode(`${mod2} = tf.keras.models.load_model("${mod1}")`)
    }

    //% block="对象名[NAME]导入onnx模型[MOD1]" blockType="command"
    //% NAME.shadow="normal"   NAME.defl="modelname"
    //% MOD1.shadow="normal"   MOD1.defl="model"
    export function onnxmod_load(parameter: any, block: any){
        let name = parameter.NAME.code
        let mod1 = parameter.MOD1.code
        Generator.addImport("import onnxruntime");
        Generator.addCode(`${name} = onnxruntime.InferenceSession(${mod1})`)
    }

    //% block="对象名[NAME]导入saved_model模型[MOD1]" blockType="command"
    //% NAME.shadow="normal"   NAME.defl="modelname"
    //% MOD1.shadow="normal"   MOD1.defl="modelpath"
    export function saved_mod_load(parameter: any, block: any){
        let name = parameter.NAME.code
        let mod1 = parameter.MOD1.code
        Generator.addCode(`${name} = tf.saved_model.load('${mod1}')`)
    }


    //% block="对象名[IMG2]将图片[IMG1]尺寸设置为[SIZ] " blockType="command"
    //% IMG2.shadow="normal"   IMG2.defl="img"
    //% IMG1.shadow="normal"   IMG1.defl="img"
    //% SIZ.shadow="normal"   SIZ.defl="(28, 28)"
    export function resize(parameter: any, block: any){
        let img2 = parameter.IMG2.code
        let img1 = parameter.IMG1.code
        let siz = parameter.SIZ.code
        Generator.addCode(`${img2} = tf.image.resize(${img1}, ${siz})`)
    }



   //% block="对象名[OBJ1][OBJ2] 解码音频数据[AUDIO] 样本数[SAMP]" blockType="command"
    //% OBJ1.shadow="normal"   OBJ1.defl=""
    //% OBJ2.shadow="normal"   OBJ2.defl=""
    //% AUDIO.shadow="normal"   AUDIO.defl="x"
    //% SAMP.shadow="normal"   SAMP.defl="16000"
    export function decode_wav(parameter: any, block: any){ 
        let obj1 = parameter.OBJ1.code
        let obj2 = parameter.OBJ2.code
        let audio = parameter.AUDIO.code
        let samp = parameter.SAMP.code
        if (obj2 != ""){
            Generator.addCode(`${obj1}, ${obj2} = tf.audio.decode_wav(${audio}, desired_channels=1, desired_samples=${samp},)`)
        }
        else {
            Generator.addCode(`${obj1}= tf.audio.decode_wav(x, desired_channels=1, desired_samples=16000,)`)

        }
   
   }

   //% block="将音频训练集[TRAIN]和验证集[VAL]转化成频谱图训练集[TRAIN2]和验证集[VAL2]" blockType="command"
    //% TRAIN.shadow="normal"   TRAIN.defl="train_ds"
    //% VAL.shadow="normal"   VAL.defl="val_ds"
    //% TRAIN2.shadow="normal"   TRAIN2.defl="train_spectrogram_ds"
    //% VAL2.shadow="normal"   VAL2.defl="val_spectrogram_ds"
    export function spectrogram(parameter: any, block: any){ 
        let train = parameter.TRAIN.code
        let val = parameter.VAL.code
        let train2 = parameter.TRAIN2.code
        let val2 = parameter.VAL2.code
        Generator.addCode(`
def squeeze(audio, labels):
    audio = tf.squeeze(audio, axis=-1)
    return audio,labels
def get_spectrogram(waveform):
    spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
    print(spectrogram.shape)
    spectrogram = tf.abs(spectrogram)
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram
train_ds = ${train}.map(squeeze, tf.data.AUTOTUNE)
val_ds = ${val}.map(squeeze, tf.data.AUTOTUNE)
train_spectrogram_ds = train_ds.map(lambda audio,label: (get_spectrogram(audio), label), tf.data.AUTOTUNE)
val_spectrogram_ds = val_ds.map(lambda audio,label: (get_spectrogram(audio), label), tf.data.AUTOTUNE)
${train2} = train_spectrogram_ds.cache().shuffle(10000).prefetch(tf.data.AUTOTUNE)
${val2} = val_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)
norm_layer =  keras.layers.Normalization()
norm_layer.adapt(data=train_spectrogram_ds.map(map_func=lambda spec, label: spec))
    
        `)
   
   }

    //% block="对象名[OBJ] 将单个音频数据[DATA]转化成频谱图" blockType="command"
    //% OBJ.shadow="normal"   OBJ.defl="spectrogram"
    //% DATA.shadow="normal"   DATA.defl="waveform"
    export function spectrogram_single(parameter: any, block: any){ 
        let obj = parameter.OBJ.code
        let data = parameter.DATA.code
        Generator.addCode(`
x = tf.squeeze(${data}, axis=-1)
spectrogram = tf.signal.stft(x, frame_length=255, frame_step=128)
spectrogram = tf.abs(spectrogram)
spectrogram = spectrogram[..., tf.newaxis]
x = spectrogram
${obj} = x[tf.newaxis,...]
        `)
   
   }



    //% block="将预测结果[PRE]映射到分类[CLA]" blockType="reporter"
    //% PRE.shadow="normal"   PRE.defl="predictions"
    //% CLA.shadow="normal"   CLA.defl="class_names"
    export function map(parameter: any, block: any){ 
        let pre = parameter.PRE.code
        let cla = parameter.CLA.code
        Generator.addCode(`str(${cla}[${pre}.argmax()])`)
   
   }



    //% block="将预测结果[PRE]转化为numpy数组并映射到分类[CLA]" blockType="reporter"
    //% PRE.shadow="normal"   PRE.defl="predictions"
    //% CLA.shadow="normal"   CLA.defl="class_names"
    export function map_numpy(parameter: any, block: any){ 
        let pre = parameter.PRE.code
        let cla = parameter.CLA.code
        Generator.addCode(`str(${cla}[${pre}.numpy().argmax()])`)
   
   }

   //% block="将数据[PRE2]传递给模型[MOD3]" blockType="reporter"
    //% PRE2.shadow="normal"   PRE2.defl="data_name"
    //% MOD3.shadow="normal"   MOD3.defl="model"
    export function predict_model(parameter: any, block: any){ 
        let pre2 = parameter.PRE2.code
        let mod3 = parameter.MOD3.code
        Generator.addCode(`${mod3}(${pre2}) `);
   
   }


    //% block="对象名[PRE2]用模型[MOD3]预测图片[IMG7]" blockType="command"
    //% PRE2.shadow="normal"   PRE2.defl="predictions"
    //% IMG7.shadow="normal"   IMG7.defl="img"
    //% MOD3.shadow="normal"   MOD3.defl="model"
    export function predict(parameter: any, block: any){ 
        let pre2 = parameter.PRE2.code
        let img7 = parameter.IMG7.code
        let mod3 = parameter.MOD3.code
        Generator.addCode(`${pre2} = ${mod3}.predict(${img7}) `);
   
   }

   //% block="对象名[PRE2]用ONNX模型[MOD3]预测图片数组[IMG7]" blockType="command"
    //% PRE2.shadow="normal"   PRE2.defl="predictions"
    //% IMG7.shadow="normal"   IMG7.defl="img"
    //% MOD3.shadow="normal"   MOD3.defl="model"
    export function onnx_predict(parameter: any, block: any){ 
        let pre2 = parameter.PRE2.code
        let img7 = parameter.IMG7.code
        let mod3 = parameter.MOD3.code
        Generator.addCode(`
input_name = ${mod3}.get_inputs()[0].name
output_name = ${mod3}.get_outputs()[0].name
${pre2} = ${mod3}.run([output_name], {input_name: ${img7}.astype(np.float32)})[0]
        `);
   
   }

    //% block="将数据[OBJ]独立热编码的结果返回变量[VALVE]中" blockType="command" 
    //% VALVE.shadow="normal" VALVE.defl="y" 
    //% OBJ.shadow="normal" OBJ.defl="y" 
    export function Sklearn_to_categorical(parameter: any, block: any){ 
        let obj=parameter.OBJ.code; 
        let value=parameter.VALVE.code; 
        Generator.addCode(`${value} = keras.utils.to_categorical(${obj},num_classes=None)`) 
            } 

    //% block="输出[PRE3]预测结果" blockType="command"
    //% PRE3.shadow="normal"   PRE3.defl="predictions"
    export function predict_output(parameter: any, block: any){ 
        let pre3 = parameter.PRE3.code
        Generator.addCode(`${pre3} = tf.nn.softmax(${pre3})`)
   
   }


    //% block="为模型[OBJ]添加[VALUE]层" blockType="command"
    //% OBJ.shadow="normal" OBJ.defl="model"
    //% VALUE.shadow="normal" VALUE.defl=""
    export function model_add_layer(parameter: any, block: any){
        let obj=parameter.OBJ.code;  
        let value=parameter.VALUE.code;
        Generator.addCode(`${obj}.add(${value})`)
    }

    //% block="为模型[OBJ]频谱图输入层" blockType="command"
    //% OBJ.shadow="normal" OBJ.defl="model"
    export function init_input_layer(parameter: any, block: any){
        let obj=parameter.OBJ.code;  
        Generator.addCode(`
${obj}.add(keras.layers.Input(shape=(124,129,1)))
${obj}.add(keras.layers.Resizing(32,32))
${obj}.add(norm_layer)
        `)

    }



    //% block="Dense层，神经元个数为[VALUE] 激活函数为[MET]" blockType="reporter"
    //% VALUE.shadow="normal" VALUE.defl="10"
    //% MET.shadow="dropdown" MET.options="MET"
    export function init_dense_layer(parameter: any, block: any){
        let value=parameter.VALUE.code;
        let met=parameter.MET.code; 
        if (met != ""){
            Generator.addCode(`keras.layers.Dense(${value}, activation="${met}")`)
        }
        else {
            Generator.addCode(`keras.layers.Dense(${value})`)

        }
        
    }

    //% block="卷积层，卷积核数量[COUNT]，激活函数为[MET]，输入数据格式[INPUT]" blockType="reporter"
    //% COUNT.shadow="dropdown" COUNT.options="COUNT"
    //% MET.shadow="dropdown" MET.options="MET"
    //% INPUT.shadow="dropdown" INPUT.options="INPUT"
    export function init_conv2d_layer(parameter: any, block: any){
        let count=parameter.COUNT.code;
        let met=parameter.MET.code; 
        let input=parameter.INPUT.code;
        if (input != "none"){
            Generator.addCode(`keras.layers.Conv2D(${count}, (3,3), activation="${met}", input_shape=${input})`)
        }
        else {
            Generator.addCode(`keras.layers.Conv2D(${count}, (3,3), activation="${met}")`)
        }
    }

    //% block="池化层" blockType="reporter"
    export function init_maxpool_layer(parameter: any, block: any){
        Generator.addCode(`keras.layers.MaxPooling2D((2,2))`)
    }

    //% block="展平层" blockType="reporter"
    export function init_flatten_layer(parameter: any, block: any){
        Generator.addCode(`keras.layers.Flatten()`)

    
    }



    //% block="模型[VALUE]的结构" blockType="reporter"
    //% VALUE.shadow="normal" VALUE.defl="model"
    export function model_summary(parameter: any, block: any){
        let value=parameter.VALUE.code;
        Generator.addCode(`${value}.summary()`)

    }

    //% block="设置模型[Model]参数，优化器[OPT],损失函数[LOSS], 评价函数[METRICS]"
    //% Model.shadow="normal" Model.defl="model"
    //% OPT.shadow="dropdown" OPT.options="OPT"
    //% LOSS.shadow="dropdown" LOSS.options="LOSS"
    //% METRICS.shadow="dropdown" METRICS.options="METRICS"
    export function model_setting(parameter: any, block: any){
        let m=parameter.Model.code;
        let o=parameter.OPT.code;
        let l=parameter.LOSS.code;
        let metrics=parameter.METRICS.code;
        Generator.addCode(`${m}.compile(optimizer="${o}", loss=${l}, metrics=['${metrics}'])`)

    }

    //% block="训练模型[OBJECT] 训练数据集[Y] 训练次数[C] 验证数据集[V]" blockType="command"
    //% OBJECT.shadow="normal" OBJECT.defl="model"
    //% Y.shadow="normal" Y.defl="train_generator"
    //% C.shadow="normal" C.defl="3"
    //% V.shadow="normal" V.defl="val_generator"
    export function Sklearn_initread2(parameter: any, block: any) {
        let obj=parameter.OBJECT.code;  
        let y=parameter.Y.code;  
        let c=parameter.C.code; 
        let v=parameter.V.code;
        if (v!= ""){
            Generator.addCode(`${obj}.fit(${y}, epochs=${c}, batch_size=32, validation_data=${v})`) 
        }
        else {
            Generator.addCode(`${obj}.fit(${y}, epochs=${c}, batch_size=32`) 
        }

    }


    //% block="以h5格式保存模型[MOD]到[OBJECT]"  blockType="command"
    //% MOD.shadow="normal" MOD.defl="model"
    //% OBJECT.shadow="normal" OBJECT.defl="model.h5"
    export function model_save_h5(parameter: any, block: any){
        let mod=parameter.MOD.code;  
        let obj=parameter.OBJECT.code;  
        Generator.addCode(`${mod}.save("${obj}")`) 
    }

    //% block="以saved_model格式保存模型[MOD]到[OBJECT]"  blockType="command"
    //% MOD.shadow="normal" MOD.defl="model"
    //% OBJECT.shadow="normal" OBJECT.defl="saved_model"
    export function model_save(parameter: any, block: any){
        let mod=parameter.MOD.code;  
        let obj=parameter.OBJECT.code;  
        Generator.addCode(`tf.saved_model.save(${mod}, '${obj}')`) 
    }


    //% block="将saved_model模型[MOD]转化为ONNX模型[ONNX]"  blockType="command"
    //% MOD.shadow="normal" MOD.defl="saved_model"
    //% ONNX.shadow="normal" ONNX.defl="model.onnx"
    export function model_trans(parameter: any, block: any){
        let mod=parameter.MOD.code;  
        let onnx=parameter.ONNX.code;  
        Generator.addImport("import subprocess");
        Generator.addCode(`subprocess.run("python -m tf2onnx.convert --saved-model ${mod} --output ${onnx}", shell=True)`) 
    }


    //% block="将数据集[SJJ]的音频文件转化为频谱图并保存到文件夹[DOC]" blockType="command"
    //% SJJ.shadow="normal" SJJ.defl="train_spectrogram_ds"
    //% DOC.shadow="normal" DOC.defl="spectrogram_images"
    export function spec(parameter: any, block: any){
        let sjj=parameter.SJJ.code;
        let doc=parameter.DOC.code;
        Generator.addImport("import os");
        Generator.addImport("import matplotlib.pyplot as plt");
        Generator.addImport("import numpy as np");
        Generator.addCode(`
# 保存频谱图为 JPG 文件
def save_spectrogram_as_image(spectrogram, label, index):
    save_dir = "${doc}"
    os.makedirs(save_dir, exist_ok=True)
    
    # 将频谱图转换为 NumPy 数组
    spectrogram_np = spectrogram.numpy()
    
    # 构建文件名
    label_name = label_names[label]
    filename = f"{label_name}_{index}.jpg"
    filepath = os.path.join(save_dir, filename)
    
    # 保存为 JPG 文件
    plt.imsave(filepath, np.squeeze(spectrogram_np), cmap='viridis')

# 遍历数据集并保存频谱图
for spectrogram_batch, label_batch in ${sjj}:
    for i in range(len(spectrogram_batch)):
        save_spectrogram_as_image(spectrogram_batch[i], label_batch[i], i)
        
        `)
    }




    //% block="获取数据[OBJ]的格式" blockType="reporter"
    //% OBJ.shadow="normal" OBJ.defl=""
    export function data_shape(parameter: any, block: any){
        let obj=parameter.OBJ.code;
        Generator.addCode(`${obj}.shape`)
    }

    //% block="创建文件夹[FLO]来保存数据 " blockType="command"
    //% FLO.shadow="normal"   FLO.defl="flodername"
    export function floder(parameter: any, block: any){
        let flo = parameter.FLO.code
        Generator.addImport("import os");
        Generator.addCode(`
if not os.path.exists(${flo}):
    os.makedirs(${flo})`);
    }

    //% block="将路径[PATH1]和路径[PATH2]连接起来 " blockType="reporter"
    //% PATH1.shadow="normal"   PATH1.defl="path_name"
    //% PATH2.shadow="normal"   PATH2.defl="path_name"
    export function path(parameter: any, block: any){
        let path1 = parameter.PATH1.code
        let path2 = parameter.PATH2.code
        Generator.addImport("import os");
        Generator.addCode(`os.path.join(${path1}, ${path2})`);
    }

    //% block="对象名[PATH1] 将文件路径[PATH2]转换成Path对象 " blockType="command"
    //% PATH1.shadow="normal"   PATH1.defl="path_name"
    //% PATH2.shadow="normal"   PATH2.defl="path_name"
    export function pathlib(parameter: any, block: any){
        let path1 = parameter.PATH1.code
        let path2 = parameter.PATH2.code
        Generator.addImport("import pathlib");
        Generator.addCode(`${path1} = pathlib.Path(${path2})`);
    }



}
