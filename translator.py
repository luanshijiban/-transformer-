import os
import torch
from transformers import MarianMTModel, MarianTokenizer
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from tkinter.font import Font
import threading
import queue
import time
from PIL import Image, ImageTk
import re
import sys
import traceback

class ColorfulTranslatorApp:
    def __init__(self, master):
        self.master = master
        self.setup_window()
        self.create_styles()
        
        # 创建主框架
        self.main_frame = ttk.Frame(self.master, padding="20", style="LightBg.TFrame")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 初始化变量 - 必须在create_widgets之前
        self.auto_translate_var = tk.BooleanVar(value=True)  # 默认开启自动翻译
        self.direction_var = tk.StringVar(value="EN")
        self.model_choice = tk.StringVar(value="1")
        
        # 创建界面组件
        self.create_widgets()
        
        # 模型缓存
        self.model_cache = {}
        self.translation_queue = queue.Queue()
        self.translation_thread = None
        self.is_processing = False
        
        # 设置性能选项 - 自动使用GPU和FP16（如果可用）
        self.use_gpu = torch.cuda.is_available()
        self.use_fp16 = torch.cuda.is_available()
        
        # 确保最基本的tokenizer初始化
        self.ensure_basic_tokenizer()
        
        # 初始化模型
        self.load_model()
        
        # 启动后台翻译线程
        self.start_background_worker()
    
    def setup_window(self):
        """配置主窗口"""
        self.master.title("🌈 基于Transformer模型的英汉双向神经机器翻译系统")
        self.master.geometry("1280x768")
        self.master.minsize(800, 600)
        self.master.configure(bg="#f0f5ff")  # 使用淡蓝色作为背景
        
        # 设置窗口图标（如果有）
        try:
            self.master.iconbitmap("translator_icon.ico")
        except:
            pass
        
        # 使窗口居中
        self.center_window()
    
    def center_window(self):
        """使窗口居中显示"""
        self.master.update_idletasks()
        width = self.master.winfo_width()
        height = self.master.winfo_height()
        x = (self.master.winfo_screenwidth() // 2) - (width // 2)
        y = (self.master.winfo_screenheight() // 2) - (height // 2)
        self.master.geometry(f"+{x}+{y}")
    
    def create_styles(self):
        """创建自定义样式"""
        self.style = ttk.Style()
        
        # 主主题
        self.style.theme_use("clam")
        
        # 自定义字体
        self.title_font = Font(family="Microsoft YaHei", size=18, weight="bold")
        self.subtitle_font = Font(family="Microsoft YaHei", size=12)
        self.text_font = Font(family="Consolas", size=11)
        
        # 配置颜色主题
        self.colors = {
            "primary": "#87CEEB",
            "secondary": "#a29bfe",
            "accent": "#fd79a8",
            "background": "white",
            "text": "#2d3436",
            "success": "#00b894",
            "warning": "#fdcb6e",
            "error": "#d63031",
            "light_bg": "#f9f9f9"
        }
        
        # 基础样式
        self.style.configure(".", 
                           background=self.colors["background"],
                           foreground=self.colors["text"])
        
        # 半透明标签框架样式
        self.style.configure("LightBg.TFrame",
                           background=self.colors["light_bg"])
        
        # 标签样式
        self.style.configure("TLabel", 
                           font=self.subtitle_font,
                           background=self.colors["background"])
        
        # 按钮样式
        self.style.configure("TButton", 
                           font=self.subtitle_font,
                           padding=8,
                           relief="flat")
        
        # 强调按钮样式
        self.style.configure("Primary.TButton", 
                           background=self.colors["primary"],
                           foreground="white")
        self.style.map("Primary.TButton",
                     background=[("active", self.colors["secondary"]), 
                               ("disabled", "#dfe6e9")])
        
        # 次要按钮样式
        self.style.configure("Secondary.TButton", 
                           background=self.colors["secondary"],
                           foreground="white")
        
        # 标签框架样式
        self.style.configure("TLabelframe", 
                           background=self.colors["light_bg"],
                           borderwidth=2,
                           relief="solid")
        self.style.configure("TLabelframe.Label", 
                           background=self.colors["light_bg"],
                           font=self.subtitle_font)
    
    def create_widgets(self):
        """创建界面控件"""
        # 主要文本框架
        self.create_text_areas()
        
        # 创建简化的控制面板
        self.create_simplified_controls()
        
        # 创建直接按钮
        self.create_direct_buttons()
        
        # 创建状态栏
        self.create_status_bar()
        
        # 字体设置
        self.set_widgets_font()
        
        # 设置滚动条同步
        self.synchronize_scrolling()
        
        # 将所有控件提到前台
        self.bring_widgets_to_front()
        
        # 绑定输入变化检测
        self.input_text.bind("<<Modified>>", self.on_input_changed)
    
    def create_text_areas(self):
        """创建输入输出文本区域"""
        # 文本区域框架
        text_frame = tk.Frame(self.main_frame, bg='#f0f0f0')
        text_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # 左右分栏
        left_frame = tk.Frame(text_frame, bg='#f0f0f0')
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        right_frame = tk.Frame(text_frame, bg='#f0f0f0')
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 输入文本标签和区域
        input_label = tk.Label(
            left_frame, 
            text="输入文本", 
            bg='#f0f0f0', 
            font=("Microsoft YaHei", 12, "bold")
        )
        input_label.pack(pady=(0, 5), anchor='w')
        
        # 输入文本区域带滚动条
        input_scroll_y = tk.Scrollbar(left_frame)
        input_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        
        input_scroll_x = tk.Scrollbar(left_frame, orient=tk.HORIZONTAL)
        input_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.input_text = tk.Text(
            left_frame,
            wrap=tk.WORD,
            yscrollcommand=input_scroll_y.set,
            xscrollcommand=input_scroll_x.set,
            height=15,
            width=50,
            bg='white',
            relief=tk.SUNKEN,
            bd=2
        )
        self.input_text.pack(fill=tk.BOTH, expand=True, padx=(0, 10))
        
        input_scroll_y.config(command=self.input_text.yview)
        input_scroll_x.config(command=self.input_text.xview)
        
        # 输出文本标签和区域
        output_label = tk.Label(
            right_frame, 
            text="翻译结果", 
            bg='#f0f0f0',
            font=("Microsoft YaHei", 12, "bold")
        )
        output_label.pack(pady=(0, 5), anchor='w')
        
        # 输出文本区域带滚动条
        output_scroll_y = tk.Scrollbar(right_frame)
        output_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        
        output_scroll_x = tk.Scrollbar(right_frame, orient=tk.HORIZONTAL)
        output_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.output_text = tk.Text(
            right_frame,
            wrap=tk.WORD,
            yscrollcommand=output_scroll_y.set,
            xscrollcommand=output_scroll_x.set,
            height=15,
            width=50,
            bg='white',
            relief=tk.SUNKEN,
            bd=2
        )
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=(10, 0))
        
        self.output_text.config(state=tk.DISABLED)  # 初始设为只读
        output_scroll_y.config(command=self.output_text.yview)
        output_scroll_x.config(command=self.output_text.xview)
    
    def create_simplified_controls(self):
        """创建简化的控制面板"""
        control_panel = tk.Frame(self.main_frame, bg="#e6e6fa", bd=2, relief=tk.GROOVE)
        control_panel.pack(fill=tk.X, padx=20, pady=10)
        
        # 翻译方向
        direction_frame = tk.Frame(control_panel, bg="#e6e6fa")
        direction_frame.pack(side=tk.LEFT, padx=20, pady=5)
        
        direction_label = tk.Label(
            direction_frame, 
            text="翻译方向:", 
            bg="#e6e6fa",
            font=("Microsoft YaHei", 11)
        )
        direction_label.pack(side=tk.LEFT, padx=(0, 10))
        
        en_to_zh = tk.Radiobutton(
            direction_frame,
            text="英译中",
            variable=self.direction_var,
            value="EN",
            bg="#e6e6fa",
            font=("Microsoft YaHei", 11),
            command=self.on_direction_changed
        )
        en_to_zh.pack(side=tk.LEFT, padx=5)
        
        zh_to_en = tk.Radiobutton(
            direction_frame,
            text="中译英",
            variable=self.direction_var,
            value="CN",
            bg="#e6e6fa",
            font=("Microsoft YaHei", 11),
            command=self.on_direction_changed
        )
        zh_to_en.pack(side=tk.LEFT, padx=5)
        
        # 自动翻译选项 - self.auto_translate_var必须已在__init__中初始化
        self.auto_translate_check = tk.Checkbutton(
            direction_frame,
            text="自动翻译",
            variable=self.auto_translate_var,
            bg="#e6e6fa",
            font=("Microsoft YaHei", 11),
            command=self.on_auto_translate_toggle
        )
        self.auto_translate_check.pack(side=tk.LEFT, padx=20)
        
        # 模型选择
        model_frame = tk.Frame(control_panel, bg="#e6e6fa")
        model_frame.pack(side=tk.RIGHT, padx=20, pady=5)
        
        model_label = tk.Label(
            model_frame, 
            text="模型选择:", 
            bg="#e6e6fa",
            font=("Microsoft YaHei", 11)
        )
        model_label.pack(side=tk.LEFT, padx=(0, 10))
        
        small_model = tk.Radiobutton(
            model_frame,
            text="小数据量模型（测试用）",
            variable=self.model_choice,
            value="1",
            bg="#e6e6fa",
            font=("Microsoft YaHei", 11),
            command=self.on_model_changed
        )
        small_model.pack(side=tk.LEFT, padx=5)
        
        large_model = tk.Radiobutton(
            model_frame,
            text="全量数据模型（实际应用）",
            variable=self.model_choice,
            value="2",
            bg="#e6e6fa",
            font=("Microsoft YaHei", 11),
            command=self.on_model_changed
        )
        large_model.pack(side=tk.LEFT, padx=5)
    
    def create_direct_buttons(self):
        """创建操作按钮"""
        button_frame = tk.Frame(self.main_frame, bg='#f0f0f0')
        button_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # 清空按钮
        self.clear_button = tk.Button(
            button_frame,
            text="清空输入",
            command=self.clear_input,
            width=12,
            bg='#e6e6e6',
            relief=tk.RAISED,
            font=("Microsoft YaHei", 10)
        )
        self.clear_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # 复制按钮
        self.copy_button = tk.Button(
            button_frame,
            text="复制结果",
            command=self.copy_output,
            width=12,
            bg='#e6e6e6',
            relief=tk.RAISED,
            font=("Microsoft YaHei", 10)
        )
        self.copy_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # 翻译按钮
        self.translate_button = tk.Button(
            button_frame,
            text="开始翻译",
            command=self.translate_text,
            width=12,
            bg='#4CAF50',
            fg='white',
            relief=tk.RAISED,
            font=("Microsoft YaHei", 10, "bold")
        )
        self.translate_button.pack(side=tk.RIGHT, padx=5, pady=5)
    
    def create_status_bar(self):
        """创建状态栏"""
        self.status_bar = tk.Label(
            self.main_frame,
            text="就绪",
            bd=1,
            relief=tk.SUNKEN,
            anchor=tk.W,
            bg='#f0f0f0',
            font=("Microsoft YaHei", 9)
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def set_widgets_font(self):
        """设置控件字体"""
        default_font = ("Microsoft YaHei", 10)
        self.input_text.configure(font=default_font)
        self.output_text.configure(font=default_font)
    
    def synchronize_scrolling(self):
        """设置输入和输出文本框滚动同步"""
        pass  # 如果需要同步滚动可以在此实现
    
    def bring_widgets_to_front(self):
        """确保所有控件显示在前台"""
        # 提升主框架显示层级
        self.main_frame.lift()
        # 刷新窗口更新
        self.master.update_idletasks()
    
    def on_direction_changed(self):
        """翻译方向改变时的处理"""
        self.update_status("翻译方向已更改")
        # 重新加载对应方向的模型
        self.load_model()
    
    def on_model_changed(self):
        """模型选择改变时的处理"""
        self.update_status("模型选择已更改")
        # 重新加载选择的模型
        self.load_model()
    
    def get_model_key(self):
        """获取当前模型的缓存键"""
        model_choice = self.model_choice.get()
        direction = self.direction_var.get()
        use_gpu = self.use_gpu
        use_fp16 = self.use_fp16
        return f"{model_choice}_{direction}_{use_gpu}_{use_fp16}"
    
    def load_model(self):
        """加载选定的翻译模型（使用缓存）"""
        try:
            small_model_paths = {
                "EN": "./train_small/en_zh_translator_small",
                "CN": "./train_small/zh_en_translator_small"
            }
            large_model_paths = {
                "EN": "./train/en_zh_translator",
                "CN": "./train/zh_en_translator"
            }

            model_choice = self.model_choice.get()
            direction = self.direction_var.get()

            # 确定模型路径
            if model_choice == "1":  # 小数据训练模型
                model_path = small_model_paths[direction]
                model_type = "小数据量模型（测试用）"
            else:  # 大数据训练模型
                model_path = large_model_paths[direction]
                model_type = "全量数据模型（实际应用）"

            # 检查是否存在模型
            if not os.path.exists(model_path):
                error_msg = f"模型路径不存在: {model_path}"
                print(error_msg)
                messagebox.showerror("错误", error_msg)
                self.update_status("🔴 模型路径不存在")
                return False
            
            # 获取模型缓存键
            model_key = self.get_model_key()
            
            # 检查是否已缓存
            if model_key in self.model_cache:
                self.model, self.tokenizer, self.device = self.model_cache[model_key]
                direction_text = "英译中" if direction == "EN" else "中译英"
                self.update_status(f"✅ {model_type}已加载({direction_text})")
                return True
            else:
                # 更新状态
                self.update_status(f"⏳ 正在加载{model_type}...")
                self.master.update()
                
                # 设置设备
                device_name = "GPU" if self.use_gpu and torch.cuda.is_available() else "CPU"
                self.device = torch.device("cuda" if self.use_gpu and torch.cuda.is_available() else "cpu")
                
                try:
                    # 加载模型
                    self.model, self.tokenizer = self.load_model_from_path(model_path, self.device)
                    
                    # 缓存模型
                    self.model_cache[model_key] = (self.model, self.tokenizer, self.device)
                    
                    direction_text = "英译中" if direction == "EN" else "中译英"
                    self.update_status(f"✅ {model_type}加载成功({direction_text}, {device_name})")
                    return True
                except Exception as e:
                    error_msg = f"加载模型失败: {str(e)}"
                    print(error_msg)
                    messagebox.showerror("错误", error_msg)
                    self.update_status("❌ 模型加载失败")
                    return False
                
        except Exception as e:
            error_msg = f"模型加载错误: {str(e)}"
            print(error_msg)
            messagebox.showerror("错误", error_msg)
            self.update_status("❌ 模型加载错误")
            return False
    
    def load_model_from_path(self, model_path, device):
        """从指定路径加载模型"""
        # 创建进度窗口
        progress_window = tk.Toplevel(self.master)
        progress_window.title("加载模型")
        progress_window.geometry("300x100")
        progress_window.resizable(False, False)
        progress_window.transient(self.master)
        progress_window.grab_set()
        
        # 居中显示
        x = self.master.winfo_rootx() + (self.master.winfo_width() // 2) - 150
        y = self.master.winfo_rooty() + (self.master.winfo_height() // 2) - 50
        progress_window.geometry(f"+{x}+{y}")
        
        # 添加进度标签
        progress_label = ttk.Label(progress_window, text="正在加载模型，请稍候...", font=self.subtitle_font)
        progress_label.pack(pady=10)
        
        progress = ttk.Progressbar(progress_window, orient=tk.HORIZONTAL, length=250, mode='indeterminate')
        progress.pack(pady=10)
        progress.start(10)
        
        # 更新UI
        self.master.update_idletasks()
        
        try:
            # 逐步加载模型以保持UI响应
            progress_label.config(text="正在加载分词器...")
            self.master.update_idletasks()
            tokenizer = MarianTokenizer.from_pretrained(model_path)
            
            progress_label.config(text="正在加载模型...")
            self.master.update_idletasks()
            
            # 根据设备和选项加载模型
            if hasattr(self, 'quantize_var') and getattr(self, 'quantize_var').get() and device.type == 'cpu':
                # 使用8位量化来减少内存占用
                from transformers import AutoModelForSeq2SeqLM
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_path, 
                    torch_dtype=torch.int8,
                    device_map="auto",
                    load_in_8bit=True
                )
            else:
                model = MarianMTModel.from_pretrained(model_path)
                model = model.to(device)
                
                # 如果启用了FP16且支持GPU
                if self.use_fp16 and torch.cuda.is_available() and device.type == 'cuda':
                    progress_label.config(text="正在优化模型(FP16)...")
                    self.master.update_idletasks()
                    model = model.half()  # 转换为FP16
            
            return model, tokenizer
        except Exception as e:
            print(f"加载模型错误: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            # 关闭进度窗口
            progress_window.destroy()
    
    def start_background_worker(self):
        """启动后台翻译线程"""
        def worker():
            while True:
                try:
                    # 从队列中获取翻译任务
                    text, callback = self.translation_queue.get(block=True)
                    self.is_processing = True
                    
                    # 执行翻译，注意：不要调用translate_text方法，避免递归
                    result = self.perform_batch_translation(text)
                    
                    # 在主线程中更新UI
                    self.master.after(0, lambda r=result: callback(r))
                    
                    self.is_processing = False
                    self.translation_queue.task_done()
                except Exception as e:
                    self.master.after(0, lambda: self.update_status(f"🔴 翻译出错: {str(e)}"))
                    self.is_processing = False
                    self.translation_queue.task_done()
        
        # 创建并启动线程
        self.translation_thread = threading.Thread(target=worker, daemon=True)
        self.translation_thread.start()
    
    def on_input_changed(self, event=None):
        """当输入文本变化时调用"""
        # 正确方式重置修改标志
        self.input_text.edit_modified(False)
        
        if not self.auto_translate_var.get():
            return
        
        # 如果模型未加载或正在处理，则返回
        if not hasattr(self, 'model') or not hasattr(self, 'tokenizer') or self.is_processing:
            return
            
        # 获取当前输入文本
        text = self.input_text.get("1.0", tk.END).strip()
        if len(text) < 5:  # 少于5个字符不触发翻译
            return
        
        # 使用延迟翻译，避免频繁调用
        if hasattr(self, '_translate_after_id'):
            self.master.after_cancel(self._translate_after_id)
        
        # 0.8秒后执行翻译，提供更快响应
        self._translate_after_id = self.master.after(800, self.translate_text)
    
    def on_auto_translate_toggle(self):
        """自动翻译选项切换时的回调函数"""
        status = "开启" if self.auto_translate_var.get() else "关闭"
        self.update_status(f"自动翻译已{status}")
        
        # 如果开启了自动翻译，立即执行一次翻译
        if self.auto_translate_var.get():
            self.translate_text()
    
    def translate_text(self):
        """执行翻译操作"""
        input_text = self.input_text.get("1.0", tk.END).strip()
        if not input_text:
            self.update_status("请先输入文本")
            return
        
        self.update_status("正在翻译...")
        self.translate_button.config(state=tk.DISABLED)
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete("1.0", tk.END)
        
        try:
            # 获取当前的翻译方向和模型类型
            direction = self.direction_var.get()
            model_type = self.model_choice.get()
            
            # 执行翻译
            if hasattr(self, 'model') and hasattr(self, 'tokenizer'):
                result = self.perform_batch_translation(input_text)
                self.output_text.insert(tk.END, result.strip())
                self.update_status("翻译完成")
            else:
                self.output_text.insert(tk.END, "翻译模型未加载")
                self.update_status("模型未加载")
        except Exception as e:
            self.output_text.insert(tk.END, f"翻译错误: {str(e)}")
            self.update_status("翻译失败")
            print(f"翻译错误: {str(e)}")
            traceback.print_exc()
        finally:
            self.output_text.config(state=tk.DISABLED)
            self.translate_button.config(state=tk.NORMAL)
    
    def perform_batch_translation(self, input_text):
        """将长文本分批翻译"""
        if not input_text.strip():
            return ""
            
        # 将长文本分段处理以避免超出最大长度限制
        max_length = 512  # 假设最大长度为512个token
        result = ""
        
        # 简单处理：按句子分段
        sentences = re.split(r'(?<=[.!?。！？])\s+', input_text)
        batch = ""
        
        for sentence in sentences:
            if len(batch) + len(sentence) < max_length:
                batch += " " + sentence if batch else sentence
            else:
                # 翻译当前批次
                partial_result = self.perform_translation(batch)
                result += partial_result + " "
                # 开始新批次
                batch = sentence
        
        # 翻译最后一个批次
        if batch:
            partial_result = self.perform_translation(batch)
            result += partial_result
            
        return result.strip()
    
    def perform_translation(self, text):
        """执行实际的翻译操作"""
        if not text.strip():
            return ""
            
        try:
            # 使用当前模型和tokenizer进行翻译
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            # 将输入移动到正确的设备
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 生成翻译
            with torch.no_grad():
                output = self.model.generate(**inputs)
            
            # 解码结果
            result = self.tokenizer.decode(output[0], skip_special_tokens=True)
            return result
        except Exception as e:
            print(f"翻译过程中出错: {str(e)}")
            traceback.print_exc()
            return f"[翻译错误: {str(e)}]"
    
    def clear_input(self):
        """清空输入文本框"""
        self.input_text.delete("1.0", tk.END)
        self.update_status("输入已清空")
    
    def copy_output(self):
        """复制翻译结果到剪贴板"""
        output_text = self.output_text.get("1.0", tk.END).strip()
        if output_text:
            self.clipboard_clear()
            self.clipboard_append(output_text)
            self.update_status("结果已复制到剪贴板")
        else:
            self.update_status("没有可复制的内容")
    
    def update_status(self, message):
        """更新状态栏信息"""
        self.status_bar.config(text=message)
        self.master.update_idletasks()
    
    def ensure_basic_tokenizer(self):
        """确保至少有一个基本的tokenizer可用，作为后备"""
        try:
            # 添加一个简单的后备tokenizer
            from transformers import MarianTokenizer
            
            print("正在初始化基本tokenizer...")
            
            # 尝试从本地加载
            model_paths = [
                "./train_small/en_zh_translator_small",
                "./train_small/zh_en_translator_small",
                "./train/en_zh_translator",
                "./train/zh_en_translator"
            ]
            
            # 检查是否存在本地模型
            model_exists = False
            for path in model_paths:
                if os.path.exists(path):
                    print(f"找到本地模型: {path}")
                    try:
                        self.tokenizer = MarianTokenizer.from_pretrained(path)
                        print("基本tokenizer初始化成功")
                        model_exists = True
                        break
                    except Exception as e:
                        print(f"从{path}加载tokenizer失败: {str(e)}")
            
            if not model_exists:
                print("未找到本地模型，将使用最小功能集")
                # 创建一个简单的替代tokenizer
                self.tokenizer = None  # 暂无可用tokenizer
                messagebox.showwarning("警告", "未找到预训练模型，某些功能可能不可用")
                
        except Exception as e:
            print(f"初始化基本tokenizer失败: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    try:
        # 预加载部分模块以减少启动时间
        import threading
        import torch
        import os
        import sys
        import traceback
        
        print("正在启动英汉双向神经机器翻译系统...")
        print(f"Python版本: {sys.version}")
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA是否可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA设备: {torch.cuda.get_device_name(0)}")
        
        # 检查关键目录
        required_dirs = ["train_small", "train"]
        missing_dirs = [d for d in required_dirs if not os.path.exists(d)]
        if missing_dirs:
            print(f"警告：以下目录不存在 {missing_dirs}，创建这些目录")
            for d in missing_dirs:
                os.makedirs(d, exist_ok=True)
        
        # 创建和配置根窗口
        root = tk.Tk()
        
        # 捕获未处理的异常
        def show_error(exc_type, exc_value, exc_traceback):
            error_msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
            print(f"未捕获的异常:\n{error_msg}")
            try:
                from tkinter import messagebox
                messagebox.showerror("程序错误", f"发生未处理的异常:\n{str(exc_value)}\n\n请联系开发者")
            except:
                pass
        
        # 设置异常处理器
        sys.excepthook = show_error
        
        # 创建应用实例
        app = ColorfulTranslatorApp(root)
        
        # 调整窗口大小和位置
        root.update_idletasks()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        window_width = min(1280, screen_width - 100)
        window_height = min(768, screen_height - 100)
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        # 进入主循环
        print("应用已启动，关闭窗口即可退出程序")
        root.mainloop()
    except Exception as e:
        print(f"程序启动失败: {str(e)}")
        traceback.print_exc()
        
        # 显示错误消息框
        try:
            import tkinter.messagebox as msgbox
            msgbox.showerror("启动错误", f"程序启动时发生错误:\n{str(e)}\n\n请确保已安装所有必要的库和依赖。")
        except:
            # 如果连消息框都无法显示，则只能打印错误
            print("无法显示图形界面错误消息，请查看上方日志了解详情。")
        
        # 等待用户按任意键继续
        input("按回车键退出程序...")