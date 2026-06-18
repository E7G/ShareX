#region License Information (GPL v3)

/*
    ShareX - A program that allows you to take screenshots and share any file type
    Copyright (c) 2007-2026 ShareX Team

    This program is free software; you can redistribute it and/or
    modify it under the terms of the GNU General Public License
    as published by the Free Software Foundation; either version 2
    of the License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

    Optionally you can also view the license at <http://www.gnu.org/licenses/>.
*/

#endregion License Information (GPL v3)

using ShareX.HelpersLib;
using RapidOcrNet;
using SkiaSharp;
using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace ShareX
{
    public static class OCRHelper
    {
        private static readonly OCRLanguage[] Languages =
        [
            new OCRLanguage("Chinese (Simplified)", "zh-CN"),
            new OCRLanguage("English", "en"),
            new OCRLanguage("Japanese", "ja"),
            new OCRLanguage("Korean", "ko")
        ];

        public static OCRLanguage[] AvailableLanguages
        {
            get
            {
                return Languages;
            }
        }

        public static void ThrowIfNotSupported()
        {
        }

        public static async Task<string> OCR(Bitmap bmp, string languageTag = "en", float scaleFactor = 1f, bool singleLine = false)
        {
            ThrowIfNotSupported();

            scaleFactor = Math.Max(scaleFactor, 1f);

            return await Task.Run(async () =>
            {
                using (Bitmap bmpScaled = ImageHelpers.ScaleImageFast(bmp, scaleFactor))
                {
                    return OCRInternal(bmpScaled, languageTag, singleLine);
                }
            });
        }

        private static string OCRInternal(Bitmap bmp, string languageTag, bool singleLine = false)
        {
            using RapidOcr ocr = new RapidOcr();
            InitModels(ocr, languageTag);

            using (MemoryStream stream = new MemoryStream())
            {
                bmp.Save(stream, ImageFormat.Png);
                stream.Position = 0;

                using SKBitmap skBitmap = SKBitmap.Decode(stream);
                OcrResult ocrResult = ocr.Detect(skBitmap, RapidOcrOptions.Default);

                string separator = singleLine ? " " : Environment.NewLine;
                string[] lines = ocrResult.TextBlocks.Select(x => x.Text).Where(x => !string.IsNullOrWhiteSpace(x)).ToArray();

                return string.Join(separator, lines);
            }
        }

        private static void InitModels(RapidOcr ocr, string languageTag)
        {
            if (languageTag.StartsWith("zh", StringComparison.OrdinalIgnoreCase))
            {
                string modelsDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "models", "v5");
                string detPath = Path.Combine(modelsDir, "ch_PP-OCRv5_mobile_det.onnx");
                string clsPath = Path.Combine(modelsDir, "ch_ppocr_mobile_v2.0_cls_infer.onnx");
                string recPath = Path.Combine(modelsDir, "ch_PP-OCRv5_rec_mobile.onnx");
                string keysPath = Path.Combine(modelsDir, "ppocrv5_dict.txt");

                if (File.Exists(detPath) && File.Exists(clsPath) && File.Exists(recPath) && File.Exists(keysPath))
                {
                    ocr.InitModels(detPath, clsPath, recPath, keysPath);
                    return;
                }
            }

            ocr.InitModels();
        }
    }
}
