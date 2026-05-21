# 通道
## 滚转
### 单位
- deg
- deg/s

### 角度环节
- 角度指令与角度状态的误差，经过控制器pid(5,0,0)，得到角速度指令

### 角速度环节
- 角速度指令与角速度状态的误差，经过控制器100/7*pid(0.1,0.05,0)*(1/(2*pi*2)*s+1)/(1/(2*pi*20)*s+1)，得到通道控制量servo_roll

## 俯仰
### 单位
- deg
- deg/s

### 角度环节
- 角度指令与角度状态的误差，经过控制器pid(5,0,0)，得到角速度指令

### 角速度环节
- 角速度指令与角速度状态的误差，经过控制器100/7*pid(0.1,0.05,0)*(1/(2*pi*2)*s+1)/(1/(2*pi*20)*s+1)，得到通道控制量servo_pitch

## 偏航
### 单位
- deg
- deg/s

### 角度环节
- 角度指令与角度状态的误差，经过控制器pid(2,0,0)，得到角速度指令

### 角速度环节
- 角速度指令与角速度状态的误差，经过控制器100/10*pid(0.3,0.1,0)，得到通道控制量servo_yaw

## 油门
### 单位
- m
- m/s
- m/ss

### 高度环节
- 高度指令与高度状态的误差，经过控制器pid(0.7,0,0)，得到速度指令

### 速度环节
- 速度指令与速度状态的误差，经过控制器pid(2,0,0)，得到加速度指令

### 加速度环节
- 加速度指令与加速度状态的误差，经过控制器100*pid(0.1,2,0)，得到油门通道控制量

### 倾角补偿
- 油门修正控制量servo_thro = 油门通道控制量/cos(pitch)/cos(roll)

## 水平速度
- 水平速度指令与水平速度状态，经过控制器pid(1,0.02,0)，得到水平加速度指令
- 水平加速度转到机体坐标系，映射到{滚转角指令，俯仰角指令}

## 水平位置
- 水平位置指令与水平位置状态，经过控制器pid(1,0,0)，得到水平速度指令

# 分配
- M1 = servo_thro - servo_roll + servo_pitch - servo_yaw
- M2 = servo_thro + servo_roll + servo_pitch + servo_yaw
- M3 = servo_thro + servo_roll - servo_pitch - servo_yaw
- M4 = servo_thro - servo_roll - servo_pitch + servo_yaw
- M5 = servo_thro - servo_roll + servo_pitch + servo_yaw
- M6 = servo_thro + servo_roll + servo_pitch - servo_yaw
- M7 = servo_thro + servo_roll - servo_pitch + servo_yaw
- M8 = servo_thro - servo_roll - servo_pitch - servo_yaw
- 每个电机的输出行程为[0, 1000]，对应拉力范围[0, 100kg]