using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;



namespace Mammogram
{
    public partial class Form1 : Form
    {
        string currentImage;
        string previousImage;

        Tensor<float> input1;

        Tensor<float> input2;
        Bitmap img1;
        Bitmap img2;
        Bitmap resizedImgH;
        Bitmap resizedImgW;

        Bitmap bm;
        Graphics g;
        bool paint = false;
        System.Drawing.Point px, py;
        Pen p = new Pen(System.Drawing.Color.Yellow, 1);
        int index;
        int sX, sY, cX, cY;
        private List<List<System.Drawing.Point>> Polygons = new List<List<System.Drawing.Point>>();
        private List<System.Drawing.Point> NewPolygon = null;
        private System.Drawing.Point NewPoint;
        int h, w;
        int te;
        float Test;

        public Form1()
        {
            InitializeComponent();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            OpenFileDialog open = new OpenFileDialog();

            //open.Filter = "Image Files(*.jpg; *.jpeg; *.png; *.bmp)|*.jpg; *.jpeg; *.png; *.bmp";
            if (open.ShowDialog() == DialogResult.OK)
            {
                // display image in picture box 
                currentImage = open.FileName;
                img1 = new Bitmap(open.FileName);
                resizedImgH = new Bitmap(img1, new System.Drawing.Size(pictureBoxH.Width, pictureBoxH.Height));
          
                pictureBoxH.Image = img1;
                pictureBoxH.SizeMode = PictureBoxSizeMode.StretchImage;
                // image file path  
               // textBox1.Text = open.FileName;
            }



        }

        private void button2_Click(object sender, EventArgs e)
        {
            OpenFileDialog open = new OpenFileDialog();

            //open.Filter = "Image Files(*.jpg; *.jpeg; *.png; *.bmp)|*.jpg; *.jpeg; *.png; *.bmp";
            if (open.ShowDialog() == DialogResult.OK)
            {
                // display image in picture box  
                previousImage = open.FileName;
                img2 = new Bitmap(open.FileName);
                pictureBoxC.Image = new Bitmap(open.FileName);
                pictureBoxC.SizeMode = PictureBoxSizeMode.StretchImage;
                // image file path  
                // textBox1.Text = open.FileName;
            }
        }

        private void RunModel_Click(object sender, EventArgs e)
        {

            Console.WriteLine("model predicting ...");
            disText.Text = "model predicting ...";

    
            SixLabors.ImageSharp.Image<Rgb24> img1 = SixLabors.ImageSharp.Image.Load<Rgb24>(currentImage);
            SixLabors.ImageSharp.Image<Rgb24> img2 = SixLabors.ImageSharp.Image.Load<Rgb24>(previousImage);

            img1.Mutate(x =>
            {
                x.Resize(new ResizeOptions
                {
                    Size = new SixLabors.ImageSharp.Size(1024, 1024),
                    Mode = ResizeMode.Crop
                });
            });

            img2.Mutate(x =>
            {
                x.Resize(new ResizeOptions
                {
                    Size = new SixLabors.ImageSharp.Size(1024, 1024),
                    Mode = ResizeMode.Crop
                });
            });


            input1 = new DenseTensor<float>(new[] { 1, 3, 1024, 1024 });

            input2 = new DenseTensor<float>(new[] { 1, 3, 1024, 1024 });

            var mean = new[] { 0.485f, 0.456f, 0.406f };
            var stddev = new[] { 0.229f, 0.224f, 0.225f };

            img1.ProcessPixelRows(accessor =>
            {
                for (int y = 0; y < accessor.Height; y++)
                {
                    Span<Rgb24> pixelSpan = accessor.GetRowSpan(y);
                    for (int x = 0; x < accessor.Width; x++)
                    {
                        input1[0, 0, y, x] = (pixelSpan[x].R / 255f);
                        input1[0, 1, y, x] = (pixelSpan[x].G / 255f);
                        input1[0, 2, y, x] = (pixelSpan[x].B / 255f);
                    }
                }
            });

            img2.ProcessPixelRows(accessor =>
            {
                for (int y = 0; y < accessor.Height; y++)
                {
                    Span<Rgb24> pixelSpan = accessor.GetRowSpan(y);
                    for (int x = 0; x < accessor.Width; x++)
                    {
                        input2[0, 0, y, x] = (pixelSpan[x].R / 255f);
                        input2[0, 1, y, x] = (pixelSpan[x].G / 255f);
                        input2[0, 2, y, x] = (pixelSpan[x].B / 255f);
                    }
                }
            });




            var session = new Microsoft.ML.OnnxRuntime.InferenceSession("FFSb.onnx");
            IEnumerable<string> inputNames = session.InputMetadata.Keys;

            foreach (string i in inputNames)
            {
                Console.WriteLine(i);
            }

            //string InputName = inputMeta.First().Key;
            var inputs1 = new List<Microsoft.ML.OnnxRuntime.NamedOnnxValue>
            {
                Microsoft.ML.OnnxRuntime.NamedOnnxValue.CreateFromTensor("modelInput1", input1),
                Microsoft.ML.OnnxRuntime.NamedOnnxValue.CreateFromTensor("modelInput2", input2)
            };

            var inputs2 = new List<Microsoft.ML.OnnxRuntime.NamedOnnxValue>
            {
                Microsoft.ML.OnnxRuntime.NamedOnnxValue.CreateFromTensor("modelInput", input2)
           // Microsoft.ML.OnnxRuntime.NamedOnnxValue.CreateFromTensor("data", input2)
            };


            //try
            //{
            //Microsoft.ML.OnnxRuntime.IDisposableReadOnlyCollection<Microsoft.ML.OnnxRuntime.DisposableNamedOnnxValue> results = session.Run(inputs1);
            var output = session.Run(inputs1);
            //var Test = results.ToList();
            foreach (var r in output)
            {
                if (r.Name == "modelOutput")
                {

                    Test = r.AsTensor<float>().GetValue(0);
                  
                    outText.Text = Test.ToString();

                    if (Test >= 0.5)
                    {
                        outLabel.Text = "1";
                    }
                    else
                    {
                        outLabel.Text = "0";

                    }

                }

            }

            //Console.WriteLine(Test.ToString());
            //outText.Text = Test.ToString();
            //}
            //catch
            //{
            // outText.Text = "incorrect input";
            // }

            Console.WriteLine("Prediction completed");
            disText.Text = "Prediction completed";
        }

        private void label3_Click(object sender, EventArgs e)
        {

        }

        private void textBox2_TextChanged(object sender, EventArgs e)
        {

        }



        private void Main_pictureBox_MouseClick(object sender, MouseEventArgs e)
        {
            //label11.Text = "X = " + e.X + " ; Y = " + e.Y;
        }

 

        private void Main_pictureBox_Click(object sender, EventArgs e)
        {

        }


        private void picOne_MouseDown(object sender, MouseEventArgs e)
        {

      
            py = e.Location;

            cX = e.X;
            cY = e.Y;
           
        }

        // Picture box


        // End



        private void picOne_MouseMove(object sender, MouseEventArgs e)
        {

            sX = e.X;
            sY = e.Y;
            outText.Text = sX.ToString();
            textBox2.Text = cX.ToString();

        }

        private void label4_Click(object sender, EventArgs e)
        {

        }

        private void textBox1_TextChanged(object sender, EventArgs e)
        {

        }

        private void SaveRusults_Click(object sender, EventArgs e)
        {

        }

        private void picOne_Mouseup(object sender, MouseEventArgs e)
        {
            
        
        }

        private void pictwo_MouseDown(object sender, MouseEventArgs e)
        {


            py = e.Location;

            cX = e.X;
            cY = e.Y;

        }

        // Picture box


        // End



        private void pictwo_MouseMove(object sender, MouseEventArgs e)
        {

            sX = e.X;
            sY = e.Y;
            //  label11.Text = "X = " + e.X + " ; Y = " + e.Y;
        }



        private void annotation_Click(object sender, EventArgs e)
        {

            Form2 popWindow = new Form2();
            popWindow.ShowDialog();

        }

        private void annotation2_Click(object sender, EventArgs e)
        {
            for (int x = cX; x <= sX; x++)
            {
                for (int y = cY; y <= cY; y++)
                {
                    input2[0, 0, y, x] = 0;
                    input2[0, 1, y, x] = 0;
                    input2[0, 2, y, x] = 0;
                }
            }
            System.Drawing.Color newcol = System.Drawing.Color.FromArgb(0, 0, 0);
            for (int x = cX; x <= sX; x++)
            {
                for (int y = cY; y <= cY; y++)
                {

                    img2.SetPixel(x, y, newcol);
                }
            }

            pictureBoxC.Image = img2;
            pictureBoxC.SizeMode = PictureBoxSizeMode.StretchImage;

        }
    }
}
