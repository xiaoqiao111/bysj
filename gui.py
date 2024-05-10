import tkinter as tk
import tkinter.filedialog as fd

def recognoze():

    label1.destroy()
    label.config(text="识别结果："+"health/depression")
# 创建一个按钮小部件用于选择文件夹里的文件
def select_file():
    file_path = fd.askopenfilename()
    # 在这里处理所选择的文件路径，可以将路径显示在界面上的一个Label中

# 创建一个按钮小部件用于设置文件保存路径
def set_save_path():
    save_path = fd.asksaveasfilename()  # 弹出文件保存路径选择框
    # 在这里处理文件保存路径，可以将路径显示在界面上的一个Label中

# 创建主窗口
root = tk.Tk()
root.title("depression recognition")
root.config(background ="#467578")

# 创建一个标签小部件
label = tk.Label(root, text="抑郁症识别系统")
# 将标签放入主窗口
label.place(x = 20,y = 1)

# 创建一个标签小部件
label1 = tk.Label(root, anchor="ne", text="蒲晓巧")
# 将标签放入主窗口
label1.place(x = 150,y = 30)



# 创建一个按钮小部件
button = tk.Button(root, text="识别", command=recognoze)
# 将按钮放入主窗口
button.place(x = 80,y = 150)





root.mainloop()