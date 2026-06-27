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

using Avalonia.Media.Imaging;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using ShareX.ImageEditor.Core.BackgroundRemoval;
using ShareX.ImageEditor.Hosting;
using ShareX.ImageEditor.Presentation.Rendering;
using ShareX.ImageEditor.Presentation.Theming;
using SkiaSharp;
using System.Collections.ObjectModel;
using System.Diagnostics;

namespace ShareX.ImageEditor.Presentation.ViewModels;

public sealed partial class BackgroundRemoverViewModel : ViewModelBase, IDisposable
{
    private readonly BackgroundRemovalService _backgroundRemovalService = new();
    private SKBitmap? _sourceBitmap;

    public BackgroundRemoverViewModel(string? modelsFolder)
    {
        ModelsFolder = modelsFolder;
        RefreshModels();
    }

    public ObservableCollection<BackgroundRemovalModel> AvailableModels { get; } = [];

    public IReadOnlyList<BackgroundRemovalDevice> AvailableDevices { get; } = Enum.GetValues<BackgroundRemovalDevice>();

    public string? ModelsFolder { get; }

    [ObservableProperty]
    private BackgroundRemovalDevice _selectedDevice = BackgroundRemovalDevice.Auto;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasSelectedModel))]
    [NotifyPropertyChangedFor(nameof(CanRemoveBackground))]
    [NotifyCanExecuteChangedFor(nameof(RemoveBackgroundCommand))]
    private BackgroundRemovalModel? _selectedModel;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasImage))]
    [NotifyPropertyChangedFor(nameof(CanRemoveBackground))]
    [NotifyCanExecuteChangedFor(nameof(RemoveBackgroundCommand))]
    private string? _imagePath;

    [ObservableProperty]
    private Bitmap? _previewImage;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CanRemoveBackground))]
    [NotifyCanExecuteChangedFor(nameof(BrowseImageCommand))]
    [NotifyCanExecuteChangedFor(nameof(RefreshModelsCommand))]
    [NotifyCanExecuteChangedFor(nameof(OpenModelsFolderCommand))]
    [NotifyCanExecuteChangedFor(nameof(RemoveBackgroundCommand))]
    private bool _isProcessing;

    public Func<string, Task<string?>>? SelectImageFileRequested { get; set; }

    public bool HasImage => !string.IsNullOrEmpty(ImagePath);

    public bool HasSelectedModel => SelectedModel != null;

    public bool CanRemoveBackground => HasImage && HasSelectedModel && !IsProcessing;

    [RelayCommand(CanExecute = nameof(CanRefreshModels))]
    private void RefreshModels()
    {
        string? selectedModelPath = SelectedModel?.FilePath;

        AvailableModels.Clear();
        SelectedModel = null;

        if (string.IsNullOrWhiteSpace(ModelsFolder))
        {
            return;
        }

        try
        {
            Directory.CreateDirectory(ModelsFolder);

            foreach (string modelPath in Directory.EnumerateFiles(ModelsFolder, "*.onnx", SearchOption.TopDirectoryOnly).OrderBy(Path.GetFileName))
            {
                FileInfo fileInfo = new(modelPath);
                AvailableModels.Add(new BackgroundRemovalModel
                {
                    FilePath = fileInfo.FullName,
                    FileName = fileInfo.Name,
                    FileSize = fileInfo.Length
                });
            }

            SelectedModel = AvailableModels.FirstOrDefault(model =>
                string.Equals(model.FilePath, selectedModelPath, StringComparison.OrdinalIgnoreCase))
                ?? AvailableModels.FirstOrDefault();
        }
        catch (Exception ex)
        {
            EditorServices.ReportWarning(nameof(BackgroundRemoverViewModel), "Failed to scan background removal models.", ex);
        }
    }

    [RelayCommand(CanExecute = nameof(CanOpenModelsFolder))]
    private void OpenModelsFolder()
    {
        if (string.IsNullOrWhiteSpace(ModelsFolder))
        {
            return;
        }

        try
        {
            Directory.CreateDirectory(ModelsFolder);

            Process.Start(new ProcessStartInfo
            {
                FileName = ModelsFolder,
                UseShellExecute = true
            });

            RefreshModels();
        }
        catch (Exception ex)
        {
            EditorServices.ReportWarning(nameof(BackgroundRemoverViewModel), "Failed to open models folder.", ex);
        }
    }

    [RelayCommand(CanExecute = nameof(CanBrowseImage))]
    private async Task BrowseImageAsync()
    {
        if (SelectImageFileRequested == null)
        {
            return;
        }

        string? filePath = await SelectImageFileRequested("Select image");
        if (string.IsNullOrEmpty(filePath))
        {
            return;
        }

        try
        {
            SKBitmap? bitmap = SKBitmap.Decode(filePath);
            if (bitmap == null)
            {
                return;
            }

            SetSourceImage(bitmap, filePath);
        }
        catch (Exception ex)
        {
            EditorServices.ReportWarning(nameof(BackgroundRemoverViewModel), $"Failed to load image '{filePath}'.", ex);
        }
    }

    [RelayCommand(CanExecute = nameof(CanRemoveBackground))]
    private async Task RemoveBackgroundAsync()
    {
        if (string.IsNullOrEmpty(ImagePath))
        {
            return;
        }

        if (SelectedModel == null)
        {
            return;
        }

        try
        {
            IsProcessing = true;
            BackgroundRemovalModel selectedModel = SelectedModel!;
            BackgroundRemovalDevice selectedDevice = SelectedDevice;
            Stopwatch stopwatch = Stopwatch.StartNew();

            SKBitmap? sourceBitmap = SKBitmap.Decode(ImagePath);
            if (sourceBitmap == null)
            {
                return;
            }

            SetSourceImage(sourceBitmap.Copy(), ImagePath);

            BackgroundRemovalResult result = await Task.Run(() =>
            {
                using (sourceBitmap)
                {
                    return _backgroundRemovalService.RemoveBackground(sourceBitmap, selectedModel, selectedDevice);
                }
            });

            SetSourceImage(result.Image, ImagePath);
            stopwatch.Stop();
            ShowNotification($"Background removed in {stopwatch.ElapsedMilliseconds} ms.", EditorIcons.ToolSmartEraser);
            string cacheStatus = result.IsSessionCached ? "cached" : "not cached";
            Debug.WriteLine(
                $"Background removal (device={selectedDevice}, execution={result.ExecutionDevice}, model={selectedModel.FileName}, {cacheStatus}): " +
                $"total={stopwatch.ElapsedMilliseconds} ms, session={result.SessionSetupMilliseconds} ms, " +
                $"preprocess={result.PreprocessingMilliseconds} ms, inference={result.InferenceMilliseconds} ms, " +
                $"postprocess={result.PostprocessingMilliseconds} ms");
        }
        catch (Exception ex)
        {
            EditorServices.ReportWarning(nameof(BackgroundRemoverViewModel), "Background removal failed.", ex);
        }
        finally
        {
            IsProcessing = false;
        }
    }

    private bool CanBrowseImage()
    {
        return !IsProcessing;
    }

    private bool CanRefreshModels()
    {
        return !IsProcessing;
    }

    private bool CanOpenModelsFolder()
    {
        return !IsProcessing && !string.IsNullOrWhiteSpace(ModelsFolder);
    }

    private void SetSourceImage(SKBitmap bitmap, string? filePath)
    {
        Bitmap preview = BitmapConversionHelpers.ToAvaloniBitmap(bitmap);

        _sourceBitmap?.Dispose();
        PreviewImage?.Dispose();

        _sourceBitmap = bitmap;
        PreviewImage = preview;
        ImagePath = filePath;
    }

    public void Dispose()
    {
        DismissNotification();
        _backgroundRemovalService.Dispose();
        _sourceBitmap?.Dispose();
        PreviewImage?.Dispose();
    }
}
