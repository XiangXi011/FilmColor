import numpy as np

# 正确打开方式，自动去掉BOM
with open('data/DVP_train_sample.csv', 'r', encoding='utf-8-sig') as f:
    lines = f.readlines()

# 步骤2：解析波长（第一行）
wavelengths = np.array([float(x) for x in lines[0].strip().split(',')])

# 步骤3：解析光谱数据（每一行为一个样本）
spectra = []
for line in lines[1:]:
    arr = [float(x) for x in line.strip().split(',')]
    spectra.append(arr)
spectra = np.array(spectra)

# 步骤4：存为npz文件，变量名需和主项目一致
np.savez('data/dvp_processed_data.npz', wavelengths=wavelengths, dvp_values=spectra)

print('转换完成！已生成 data/dvp_processed_data.npz')
