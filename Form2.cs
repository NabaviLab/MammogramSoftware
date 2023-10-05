using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Mammogram
{
    public partial class Form2 : Form


    {

        string currentImage;
        string previousImage;

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
        int late;
        public Form2()
        {
            InitializeComponent();
        }

        private void load_Click(object sender, EventArgs e)
        {
            OpenFileDialog open = new OpenFileDialog();

            //open.Filter = "Image Files(*.jpg; *.jpeg; *.png; *.bmp)|*.jpg; *.jpeg; *.png; *.bmp";
            if (open.ShowDialog() == DialogResult.OK)
            {
                // display image in picture box 
                currentImage = open.FileName;
                img1 = new Bitmap(open.FileName);
                resizedImgH = new Bitmap(img1, new System.Drawing.Size(1024, 1024));

                pictureBox1.Image = resizedImgH;
                pictureBox1.SizeMode = PictureBoxSizeMode.Normal;
                // image file path  
                // textBox1.Text = open.FileName;
            }
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
         

        }

        private void picOne_Mouseup(object sender, MouseEventArgs e)
        {
            displayText.Text = "image started to process ...";
            //sX = e.X;
            //sY = e.Y;
            cX = e.X;
            cY = e.Y;

            //  label11.Text = "X = " + e.X + " ; Y = " + e.Y;
            Console.WriteLine("image started to process ...");
            
            System.Drawing.Color newcol = System.Drawing.Color.FromArgb(0, 0, 0);

            for (int x = sX; x <= cX; x++)
            {
                for (int y = sY; y <= cY; y++)
                {
                    //temp.X = MousePosition.X / 660 * Bitmap.Width;
                    //temp.Y = Bitmap.Height - (MousePosition.Y / 440 * Bitmap.Height); // y is flipped 10000000

  

                    resizedImgH.SetPixel(x, y, System.Drawing.Color.Black);
                    
                }
            }

            Console.WriteLine("process done.");
            displayText.Text = "process done.";
            pictureBox1.Image = resizedImgH;
            pictureBox1.SizeMode = PictureBoxSizeMode.Normal;

        }
        private void save_Click(object sender, EventArgs e)
        {
            SaveFileDialog _saveFile = new SaveFileDialog();
            _saveFile.Filter = "Bitmap Image (.bmp)|*.bmp|Gif Image (.gif)|*.gif |JPEG Image (.jpeg)|*.jpeg |Png Image (.png)|*.png |Tiff Image (.tiff)|*.tiff |Wmf Image (.wmf)|*.wmf";
            if (_saveFile.ShowDialog() == DialogResult.OK)
            {
               
                resizedImgH.Save(_saveFile.FileName);
            }
        }
    }
}
