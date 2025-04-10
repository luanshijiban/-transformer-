import os
import torch
from transformers import MarianMTModel, MarianTokenizer
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from tkinter.font import Font

class ColorfulTranslatorApp:
    def __init__(self, master):
        self.master = master
        self.setup_window()
        self.create_styles()
        self.create_widgets()
        self.load_model()
        
    def setup_window(self):
        """配置主窗口"""
        self.master.title("🌈 智能翻译专家")
        self.master.geometry("800x700")
        self.master.minsize(750, 650)
        self.master.configure(bg="#f0f2f5")
        
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
            "background": "#f0f2f5",
            "text": "#2d3436",
            "success": "#00b894",
            "warning": "#fdcb6e",
            "error": "#d63031",
             "b":"#000000"
        }
        
        # 基础样式
        self.style.configure(".", 
                           background=self.colors["background"],
                           foreground=self.colors["text"])
        
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
                           background=self.colors["background"],
                           borderwidth=2,
                           relief="solid")
        self.style.configure("TLabelframe.Label", 
                           background=self.colors["background"],
                           font=self.subtitle_font,
                         )
    
    def create_widgets(self):
        """创建所有界面组件"""
        # 主框架
        self.main_frame = ttk.Frame(self.master, padding="20")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 标题区域（带彩色背景）
        self.title_frame = ttk.Frame(
            self.main_frame,
            style="TFrame"
        )
        self.title_frame.pack(fill=tk.X, pady=(0, 15))
        
        # 渐变背景标题
        self.title_label = tk.Label(
            self.title_frame,
            text="🌈 智能翻译专家",
            font=self.title_font,
            bg=self.colors["primary"],
            fg="white",
            padx=20,
            pady=10
        )
        self.title_label.pack(fill=tk.X)
        
        # 模型选择区域
        self.create_model_selection()
        
        # 方向选择区域
        self.create_direction_selection()
        
        # 输入输出区域
        self.create_io_areas()
        
        # 按钮区域
        self.create_button_area()
        
        # 状态栏
        self.create_status_bar()
    
    def create_model_selection(self):
        """创建模型选择区域"""
        self.model_frame = ttk.LabelFrame(
            self.main_frame,
            text="🔧 选择模型类型",
            padding=(15, 10)
        )
        self.model_frame.pack(fill=tk.X, pady=5)
        
        self.model_choice = tk.StringVar(value="1")
        
        # 使用彩色单选按钮
        self.small_model_radio = tk.Radiobutton(
            self.model_frame,
            text="🚀 快速测试模型 (小数据量)",
            variable=self.model_choice,
            value="1",
            command=self.load_model,
            bg=self.colors["background"],
            fg=self.colors["text"],
            activebackground=self.colors["background"],
            selectcolor=self.colors["secondary"],
            font=self.subtitle_font,
            indicatoron=1
        )
        self.small_model_radio.pack(anchor=tk.W, padx=10, pady=5)
        
        self.large_model_radio = tk.Radiobutton(
            self.model_frame,
            text="🎯 高质量翻译模型 (全量数据)",
            variable=self.model_choice,
            value="2",
            command=self.load_model,
            bg=self.colors["background"],
            fg=self.colors["text"],
            activebackground=self.colors["background"],
            selectcolor=self.colors["secondary"],
            font=self.subtitle_font,
            indicatoron=1
        )
        self.large_model_radio.pack(anchor=tk.W, padx=10, pady=5)
    
    def create_direction_selection(self):
        """创建翻译方向选择区域"""
        self.direction_frame = ttk.LabelFrame(
            self.main_frame,
            text="🌍 选择翻译方向",
            padding=(15, 10)
        )
        self.direction_frame.pack(fill=tk.X, pady=5)
        
        self.direction_var = tk.StringVar(value="EN")
        
        # 使用彩色单选按钮
        self.en_radio = tk.Radiobutton(
            self.direction_frame,
            text="🇬🇧 英文 → 中文 🇨🇳",
            variable=self.direction_var,
            value="EN",
            command=self.load_model,
            bg=self.colors["background"],
            fg=self.colors["text"],
            activebackground=self.colors["background"],
            selectcolor=self.colors["accent"],
            font=self.subtitle_font,
            indicatoron=1
        )
        self.en_radio.pack(side=tk.LEFT, padx=20, pady=5)
        
        self.cn_radio = tk.Radiobutton(
            self.direction_frame,
            text="🇨🇳 中文 → 英文 🇬🇧",
            variable=self.direction_var,
            value="CN",
            command=self.load_model,
            bg=self.colors["background"],
            fg=self.colors["text"],
            activebackground=self.colors["background"],
            selectcolor=self.colors["accent"],
            font=self.subtitle_font,
            indicatoron=1
        )
        self.cn_radio.pack(side=tk.LEFT, padx=20, pady=5)
    
    def create_io_areas(self):
        """创建输入输出区域"""
        # 输入区域
        self.input_frame = ttk.LabelFrame(
            self.main_frame,
            text="📝 输入文本",
            padding=(15, 10)
        )
        self.input_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.input_text = scrolledtext.ScrolledText(
            self.input_frame,
            height=6,
            wrap=tk.WORD,
            font=self.text_font,
            padx=10,
            pady=10,
            highlightthickness=1,
            highlightbackground=self.colors["secondary"],
            bg="white",
            fg=self.colors["text"],
            insertbackground=self.colors["primary"]
        )
        self.input_text.pack(fill=tk.BOTH, expand=True)
        
        # 输出区域
        self.output_frame = ttk.LabelFrame(
            self.main_frame,
            text="✨ 翻译结果",
            padding=(15, 10)
        )
        self.output_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.output_text = scrolledtext.ScrolledText(
            self.output_frame,
            height=6,
            wrap=tk.WORD,
            font=self.text_font,
            padx=10,
            pady=10,
            state=tk.DISABLED,
            highlightthickness=1,
            highlightbackground=self.colors["secondary"],
            bg="white",
            fg=self.colors["text"]
        )
        self.output_text.pack(fill=tk.BOTH, expand=True)
    
    def create_button_area(self):
        """创建按钮区域"""
        self.button_frame = ttk.Frame(self.main_frame)
        self.button_frame.pack(fill=tk.X, pady=15)
        
        # 使用彩色按钮
        self.translate_button = tk.Button(
            self.button_frame,
            text="🔁 开始翻译",
            command=self.translate,
            bg=self.colors["primary"],
            fg="black",
            activebackground=self.colors["secondary"],
            activeforeground="white",
            font=self.subtitle_font,
            bd=0,
            padx=20,
            pady=8,
            relief="flat",
            cursor="hand2"
        )
        self.translate_button.pack(side=tk.LEFT, padx=10)
        
        self.clear_button = tk.Button(
            self.button_frame,
            text="🧹 清空内容",
            command=self.clear_text,
            bg=self.colors["accent"],
            fg="white",
            activebackground="#e84393",
            activeforeground="white",
            font=self.subtitle_font,
            bd=0,
            padx=20,
            pady=8,
            relief="flat",
            cursor="hand2"
        )
        self.clear_button.pack(side=tk.LEFT, padx=10)
    
    def create_status_bar(self):
        """创建彩色状态栏"""
        self.status_frame = tk.Frame(
            self.main_frame,
            bg=self.colors["primary"],
            height=58
        )
        self.status_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.status_var = tk.StringVar()
        self.status_var.set("🟢 就绪")
        
        self.status_label = tk.Label(
            self.status_frame,
            textvariable=self.status_var,
            bg=self.colors["primary"],
            fg="black",
            font=("Microsoft YaHei", 13),
            anchor=tk.W,
            padx=10
        )
        self.status_label.pack(fill=tk.X)
    
    def load_model(self):
        """加载选定的翻译模型"""
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
                model_type = "🚀 快速测试模型"
            else:  # 大数据训练模型
                model_path = large_model_paths[direction]
                model_type = "🎯 高质量模型"

            if os.path.exists(model_path):
                self.model, self.tokenizer, self.device = self.load_model_from_path(model_path)
                direction_text = "🇬🇧→🇨🇳" if direction == "EN" else "🇨🇳→🇬🇧"
                self.update_status(f"🟢 {model_type}加载成功 {direction_text}")
            else:
                raise FileNotFoundError(f"模型路径不存在: {model_path}")
                
        except Exception as e:
            messagebox.showerror("错误", f"加载模型失败: {str(e)}")
            self.update_status("🔴 模型加载失败")
    
    def load_model_from_path(self, model_path):
        """从指定路径加载模型"""
        model = MarianMTModel.from_pretrained(model_path)
        tokenizer = MarianTokenizer.from_pretrained(model_path)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        return model, tokenizer, device
    
    def translate(self):
        """执行翻译并更新输出结果"""
        user_input = self.input_text.get("1.0", tk.END).strip()
        if not user_input:
            messagebox.showwarning("警告", "请输入要翻译的文本!")
            return

        try:
            self.update_status("🟡 正在翻译...")
            self.master.update()  # 更新UI显示状态
            
            translated = self.translate_text(user_input)
            
            self.output_text.config(state=tk.NORMAL)
            self.output_text.delete("1.0", tk.END)
            self.output_text.insert(tk.END, translated)
            self.output_text.config(state=tk.DISABLED)
            
            self.update_status("🟢 翻译完成")
        except Exception as e:
            messagebox.showerror("错误", f"翻译出错: {str(e)}")
            self.update_status("🔴 翻译出错")
    
    def translate_text(self, text):
        """翻译文本"""
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        translated = self.model.generate(
            **inputs,
            num_beams=5,
            no_repeat_ngram_size=2,
            length_penalty=1.0,
            max_length=100,
            min_length=1
        )
        
        translated_text = self.tokenizer.decode(translated[0], skip_special_tokens=True)
        
        # 简单的后处理
        translated_text = " ".join(translated_text.split())  # 去除多余空格
        return translated_text
    
    def clear_text(self):
        """清空输入输出框"""
        self.input_text.delete("1.0", tk.END)
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete("1.0", tk.END)
        self.output_text.config(state=tk.DISABLED)
        self.update_status("🟠 已清空内容")
    
    def update_status(self, message):
        """更新状态栏"""
        self.status_var.set(message)
        self.master.update_idletasks()

if __name__ == "__main__":
    root = tk.Tk()
    app = ColorfulTranslatorApp(root)
    root.mainloop()