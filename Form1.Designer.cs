namespace Mammogram
{
    partial class Form1
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.panel1 = new System.Windows.Forms.Panel();
            this.annotation = new System.Windows.Forms.Button();
            this.SaveRusults = new System.Windows.Forms.Button();
            this.RunModel = new System.Windows.Forms.Button();
            this.LoadCurrent = new System.Windows.Forms.Button();
            this.loadHistory = new System.Windows.Forms.Button();
            this.panel2 = new System.Windows.Forms.Panel();
            this.panel3 = new System.Windows.Forms.Panel();
            this.panel4 = new System.Windows.Forms.Panel();
            this.panel5 = new System.Windows.Forms.Panel();
            this.panel7 = new System.Windows.Forms.Panel();
            this.panel9 = new System.Windows.Forms.Panel();
            this.panel13 = new System.Windows.Forms.Panel();
            this.pictureBoxC = new System.Windows.Forms.PictureBox();
            this.panel11 = new System.Windows.Forms.Panel();
            this.label2 = new System.Windows.Forms.Label();
            this.panel8 = new System.Windows.Forms.Panel();
            this.panel12 = new System.Windows.Forms.Panel();
            this.pictureBoxH = new System.Windows.Forms.PictureBox();
            this.panel10 = new System.Windows.Forms.Panel();
            this.label1 = new System.Windows.Forms.Label();
            this.panel6 = new System.Windows.Forms.Panel();
            this.label3 = new System.Windows.Forms.Label();
            this.textBox2 = new System.Windows.Forms.TextBox();
            this.Outputlabel3 = new System.Windows.Forms.Label();
            this.outText = new System.Windows.Forms.TextBox();
            this.label4 = new System.Windows.Forms.Label();
            this.outLabel = new System.Windows.Forms.TextBox();
            this.disText = new System.Windows.Forms.TextBox();
            this.panel1.SuspendLayout();
            this.panel5.SuspendLayout();
            this.panel7.SuspendLayout();
            this.panel9.SuspendLayout();
            this.panel13.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBoxC)).BeginInit();
            this.panel11.SuspendLayout();
            this.panel8.SuspendLayout();
            this.panel12.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBoxH)).BeginInit();
            this.panel10.SuspendLayout();
            this.panel6.SuspendLayout();
            this.SuspendLayout();
            // 
            // panel1
            // 
            this.panel1.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panel1.Controls.Add(this.disText);
            this.panel1.Controls.Add(this.annotation);
            this.panel1.Controls.Add(this.SaveRusults);
            this.panel1.Controls.Add(this.RunModel);
            this.panel1.Controls.Add(this.LoadCurrent);
            this.panel1.Controls.Add(this.loadHistory);
            this.panel1.Dock = System.Windows.Forms.DockStyle.Top;
            this.panel1.Location = new System.Drawing.Point(0, 0);
            this.panel1.Name = "panel1";
            this.panel1.Size = new System.Drawing.Size(1029, 40);
            this.panel1.TabIndex = 0;
            // 
            // annotation
            // 
            this.annotation.Dock = System.Windows.Forms.DockStyle.Left;
            this.annotation.Location = new System.Drawing.Point(300, 0);
            this.annotation.Name = "annotation";
            this.annotation.Size = new System.Drawing.Size(75, 38);
            this.annotation.TabIndex = 4;
            this.annotation.Text = "Remove AnnotationH";
            this.annotation.UseVisualStyleBackColor = true;
            this.annotation.Click += new System.EventHandler(this.annotation_Click);
            // 
            // SaveRusults
            // 
            this.SaveRusults.Dock = System.Windows.Forms.DockStyle.Left;
            this.SaveRusults.Location = new System.Drawing.Point(225, 0);
            this.SaveRusults.Name = "SaveRusults";
            this.SaveRusults.Size = new System.Drawing.Size(75, 38);
            this.SaveRusults.TabIndex = 3;
            this.SaveRusults.Text = "Save Result";
            this.SaveRusults.UseVisualStyleBackColor = true;
            this.SaveRusults.Click += new System.EventHandler(this.SaveRusults_Click);
            // 
            // RunModel
            // 
            this.RunModel.Dock = System.Windows.Forms.DockStyle.Left;
            this.RunModel.Location = new System.Drawing.Point(150, 0);
            this.RunModel.Name = "RunModel";
            this.RunModel.Size = new System.Drawing.Size(75, 38);
            this.RunModel.TabIndex = 2;
            this.RunModel.Text = "Run model";
            this.RunModel.UseVisualStyleBackColor = true;
            this.RunModel.Click += new System.EventHandler(this.RunModel_Click);
            // 
            // LoadCurrent
            // 
            this.LoadCurrent.Dock = System.Windows.Forms.DockStyle.Left;
            this.LoadCurrent.Location = new System.Drawing.Point(75, 0);
            this.LoadCurrent.Name = "LoadCurrent";
            this.LoadCurrent.Size = new System.Drawing.Size(75, 38);
            this.LoadCurrent.TabIndex = 1;
            this.LoadCurrent.Text = "Load current";
            this.LoadCurrent.UseVisualStyleBackColor = true;
            this.LoadCurrent.Click += new System.EventHandler(this.button2_Click);
            // 
            // loadHistory
            // 
            this.loadHistory.Dock = System.Windows.Forms.DockStyle.Left;
            this.loadHistory.Location = new System.Drawing.Point(0, 0);
            this.loadHistory.Name = "loadHistory";
            this.loadHistory.Size = new System.Drawing.Size(75, 38);
            this.loadHistory.TabIndex = 0;
            this.loadHistory.Text = "Load history";
            this.loadHistory.UseVisualStyleBackColor = true;
            this.loadHistory.Click += new System.EventHandler(this.button1_Click);
            // 
            // panel2
            // 
            this.panel2.Dock = System.Windows.Forms.DockStyle.Left;
            this.panel2.Location = new System.Drawing.Point(0, 40);
            this.panel2.Name = "panel2";
            this.panel2.Size = new System.Drawing.Size(10, 536);
            this.panel2.TabIndex = 1;
            // 
            // panel3
            // 
            this.panel3.Dock = System.Windows.Forms.DockStyle.Right;
            this.panel3.Location = new System.Drawing.Point(1019, 40);
            this.panel3.Name = "panel3";
            this.panel3.Size = new System.Drawing.Size(10, 536);
            this.panel3.TabIndex = 2;
            // 
            // panel4
            // 
            this.panel4.Dock = System.Windows.Forms.DockStyle.Bottom;
            this.panel4.Location = new System.Drawing.Point(10, 566);
            this.panel4.Name = "panel4";
            this.panel4.Size = new System.Drawing.Size(1009, 10);
            this.panel4.TabIndex = 3;
            // 
            // panel5
            // 
            this.panel5.Controls.Add(this.panel7);
            this.panel5.Controls.Add(this.panel6);
            this.panel5.Dock = System.Windows.Forms.DockStyle.Fill;
            this.panel5.Location = new System.Drawing.Point(10, 40);
            this.panel5.Name = "panel5";
            this.panel5.Size = new System.Drawing.Size(1009, 526);
            this.panel5.TabIndex = 4;
            // 
            // panel7
            // 
            this.panel7.Controls.Add(this.panel9);
            this.panel7.Controls.Add(this.panel8);
            this.panel7.Dock = System.Windows.Forms.DockStyle.Fill;
            this.panel7.Location = new System.Drawing.Point(0, 0);
            this.panel7.Name = "panel7";
            this.panel7.Size = new System.Drawing.Size(800, 526);
            this.panel7.TabIndex = 1;
            // 
            // panel9
            // 
            this.panel9.Controls.Add(this.panel13);
            this.panel9.Controls.Add(this.panel11);
            this.panel9.Dock = System.Windows.Forms.DockStyle.Fill;
            this.panel9.Location = new System.Drawing.Point(400, 0);
            this.panel9.Name = "panel9";
            this.panel9.Size = new System.Drawing.Size(400, 526);
            this.panel9.TabIndex = 1;
            // 
            // panel13
            // 
            this.panel13.BackColor = System.Drawing.SystemColors.ActiveCaptionText;
            this.panel13.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Center;
            this.panel13.BorderStyle = System.Windows.Forms.BorderStyle.Fixed3D;
            this.panel13.Controls.Add(this.pictureBoxC);
            this.panel13.Dock = System.Windows.Forms.DockStyle.Fill;
            this.panel13.Location = new System.Drawing.Point(0, 40);
            this.panel13.Name = "panel13";
            this.panel13.Size = new System.Drawing.Size(400, 486);
            this.panel13.TabIndex = 1;
            // 
            // pictureBoxC
            // 
            this.pictureBoxC.Dock = System.Windows.Forms.DockStyle.Fill;
            this.pictureBoxC.Location = new System.Drawing.Point(0, 0);
            this.pictureBoxC.Name = "pictureBoxC";
            this.pictureBoxC.Size = new System.Drawing.Size(396, 482);
            this.pictureBoxC.TabIndex = 0;
            this.pictureBoxC.TabStop = false;
            this.pictureBoxC.MouseClick += new System.Windows.Forms.MouseEventHandler(this.pictwo_MouseDown);
            this.pictureBoxC.MouseDown += new System.Windows.Forms.MouseEventHandler(this.pictwo_MouseMove);
            // 
            // panel11
            // 
            this.panel11.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panel11.Controls.Add(this.label2);
            this.panel11.Dock = System.Windows.Forms.DockStyle.Top;
            this.panel11.Location = new System.Drawing.Point(0, 0);
            this.panel11.Name = "panel11";
            this.panel11.Size = new System.Drawing.Size(400, 40);
            this.panel11.TabIndex = 0;
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Dock = System.Windows.Forms.DockStyle.Left;
            this.label2.Font = new System.Drawing.Font("Microsoft YaHei UI", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label2.Location = new System.Drawing.Point(0, 0);
            this.label2.Name = "label2";
            this.label2.Padding = new System.Windows.Forms.Padding(0, 5, 0, 0);
            this.label2.Size = new System.Drawing.Size(187, 27);
            this.label2.TabIndex = 0;
            this.label2.Text = "Current Mammogram";
            // 
            // panel8
            // 
            this.panel8.Controls.Add(this.panel12);
            this.panel8.Controls.Add(this.panel10);
            this.panel8.Dock = System.Windows.Forms.DockStyle.Left;
            this.panel8.Location = new System.Drawing.Point(0, 0);
            this.panel8.Name = "panel8";
            this.panel8.Size = new System.Drawing.Size(400, 526);
            this.panel8.TabIndex = 0;
            // 
            // panel12
            // 
            this.panel12.BackColor = System.Drawing.SystemColors.ActiveCaptionText;
            this.panel12.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Center;
            this.panel12.BorderStyle = System.Windows.Forms.BorderStyle.Fixed3D;
            this.panel12.Controls.Add(this.pictureBoxH);
            this.panel12.Dock = System.Windows.Forms.DockStyle.Fill;
            this.panel12.Location = new System.Drawing.Point(0, 40);
            this.panel12.Name = "panel12";
            this.panel12.Size = new System.Drawing.Size(400, 486);
            this.panel12.TabIndex = 1;
            // 
            // pictureBoxH
            // 
            this.pictureBoxH.Dock = System.Windows.Forms.DockStyle.Fill;
            this.pictureBoxH.Location = new System.Drawing.Point(0, 0);
            this.pictureBoxH.Name = "pictureBoxH";
            this.pictureBoxH.Size = new System.Drawing.Size(396, 482);
            this.pictureBoxH.TabIndex = 0;
            this.pictureBoxH.TabStop = false;
            this.pictureBoxH.MouseClick += new System.Windows.Forms.MouseEventHandler(this.picOne_MouseDown);
            this.pictureBoxH.MouseDown += new System.Windows.Forms.MouseEventHandler(this.picOne_MouseMove);
            this.pictureBoxH.MouseUp += new System.Windows.Forms.MouseEventHandler(this.picOne_Mouseup);
            // 
            // panel10
            // 
            this.panel10.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panel10.Controls.Add(this.label1);
            this.panel10.Dock = System.Windows.Forms.DockStyle.Top;
            this.panel10.Location = new System.Drawing.Point(0, 0);
            this.panel10.Name = "panel10";
            this.panel10.Size = new System.Drawing.Size(400, 40);
            this.panel10.TabIndex = 0;
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Dock = System.Windows.Forms.DockStyle.Left;
            this.label1.Font = new System.Drawing.Font("Microsoft YaHei UI", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label1.Location = new System.Drawing.Point(0, 0);
            this.label1.Name = "label1";
            this.label1.Padding = new System.Windows.Forms.Padding(0, 5, 0, 0);
            this.label1.Size = new System.Drawing.Size(186, 27);
            this.label1.TabIndex = 0;
            this.label1.Text = "History Mammogram";
            // 
            // panel6
            // 
            this.panel6.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panel6.Controls.Add(this.label4);
            this.panel6.Controls.Add(this.outLabel);
            this.panel6.Controls.Add(this.label3);
            this.panel6.Controls.Add(this.textBox2);
            this.panel6.Controls.Add(this.Outputlabel3);
            this.panel6.Controls.Add(this.outText);
            this.panel6.Dock = System.Windows.Forms.DockStyle.Right;
            this.panel6.Location = new System.Drawing.Point(800, 0);
            this.panel6.Name = "panel6";
            this.panel6.Size = new System.Drawing.Size(209, 526);
            this.panel6.TabIndex = 0;
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(5, 237);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(85, 13);
            this.label3.TabIndex = 3;
            this.label3.Text = "Radiologist input";
            this.label3.Click += new System.EventHandler(this.label3_Click);
            // 
            // textBox2
            // 
            this.textBox2.Location = new System.Drawing.Point(5, 267);
            this.textBox2.Name = "textBox2";
            this.textBox2.Size = new System.Drawing.Size(197, 20);
            this.textBox2.TabIndex = 2;
            this.textBox2.TextChanged += new System.EventHandler(this.textBox2_TextChanged);
            // 
            // Outputlabel3
            // 
            this.Outputlabel3.AutoSize = true;
            this.Outputlabel3.Location = new System.Drawing.Point(5, 26);
            this.Outputlabel3.Name = "Outputlabel3";
            this.Outputlabel3.Size = new System.Drawing.Size(118, 13);
            this.Outputlabel3.TabIndex = 1;
            this.Outputlabel3.Text = "Model output prediction";
            // 
            // outText
            // 
            this.outText.Location = new System.Drawing.Point(5, 56);
            this.outText.Name = "outText";
            this.outText.Size = new System.Drawing.Size(197, 20);
            this.outText.TabIndex = 0;
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(5, 91);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(143, 13);
            this.label4.TabIndex = 5;
            this.label4.Text = "Model output prediction label";
            this.label4.Click += new System.EventHandler(this.label4_Click);
            // 
            // outLabel
            // 
            this.outLabel.Location = new System.Drawing.Point(5, 121);
            this.outLabel.Name = "outLabel";
            this.outLabel.Size = new System.Drawing.Size(197, 20);
            this.outLabel.TabIndex = 4;
            this.outLabel.TextChanged += new System.EventHandler(this.textBox1_TextChanged);
            // 
            // disText
            // 
            this.disText.Dock = System.Windows.Forms.DockStyle.Fill;
            this.disText.ForeColor = System.Drawing.Color.Blue;
            this.disText.Location = new System.Drawing.Point(375, 0);
            this.disText.Name = "disText";
            this.disText.Size = new System.Drawing.Size(652, 20);
            this.disText.TabIndex = 5;
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(1029, 576);
            this.Controls.Add(this.panel5);
            this.Controls.Add(this.panel4);
            this.Controls.Add(this.panel3);
            this.Controls.Add(this.panel2);
            this.Controls.Add(this.panel1);
            this.Name = "Form1";
            this.Text = "Form1";
            this.panel1.ResumeLayout(false);
            this.panel1.PerformLayout();
            this.panel5.ResumeLayout(false);
            this.panel7.ResumeLayout(false);
            this.panel9.ResumeLayout(false);
            this.panel13.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.pictureBoxC)).EndInit();
            this.panel11.ResumeLayout(false);
            this.panel11.PerformLayout();
            this.panel8.ResumeLayout(false);
            this.panel12.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.pictureBoxH)).EndInit();
            this.panel10.ResumeLayout(false);
            this.panel10.PerformLayout();
            this.panel6.ResumeLayout(false);
            this.panel6.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.Panel panel1;
        private System.Windows.Forms.Button LoadCurrent;
        private System.Windows.Forms.Button loadHistory;
        private System.Windows.Forms.Panel panel2;
        private System.Windows.Forms.Panel panel3;
        private System.Windows.Forms.Panel panel4;
        private System.Windows.Forms.Panel panel5;
        private System.Windows.Forms.Panel panel7;
        private System.Windows.Forms.Panel panel9;
        private System.Windows.Forms.Panel panel13;
        private System.Windows.Forms.Panel panel11;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.Panel panel8;
        private System.Windows.Forms.Panel panel12;
        private System.Windows.Forms.Panel panel10;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.Panel panel6;
        private System.Windows.Forms.Button RunModel;
        private System.Windows.Forms.PictureBox pictureBoxC;
        private System.Windows.Forms.PictureBox pictureBoxH;
        private System.Windows.Forms.Button SaveRusults;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.TextBox textBox2;
        private System.Windows.Forms.Label Outputlabel3;
        private System.Windows.Forms.TextBox outText;
        private System.Windows.Forms.Button annotation;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.TextBox outLabel;
        private System.Windows.Forms.TextBox disText;
    }
}

