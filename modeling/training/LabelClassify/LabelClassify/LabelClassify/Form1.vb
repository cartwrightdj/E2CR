Imports System.IO
Imports System.Drawing.Imaging
Imports System.Text.RegularExpressions

Public Class Form1
    Private images As List(Of String)
    Private currentIndex As Integer = -1
    Private selectedFolderPath As String
    Private labels As Dictionary(Of String, String)
    Private ReadOnly random As New Random()
    Private isErasing As Boolean = False
    Private eraserSize As Integer = 2
    Private originalImage As Bitmap
    Private maskImage As Bitmap
    Private eraserCursor As Cursor
    Private zoomFactor As Double = 1.0

    Private Sub Form1_Load(sender As Object, e As EventArgs) Handles MyBase.Load
        images = New List(Of String)()
        labels = New Dictionary(Of String, String)()
        ' Set PictureBox properties
        PictureBox1.SizeMode = PictureBoxSizeMode.Zoom
        PictureBox2.SizeMode = PictureBoxSizeMode.Zoom
        ' Set the form to capture key presses
        Me.KeyPreview = True
        ' Initialize the eraser cursor
        UpdateEraserCursor()
    End Sub

    Private Sub UpdateEraserCursor()
        Dim cursorSize As Integer = eraserSize
        Dim eraserBitmap As New Bitmap(cursorSize, cursorSize)
        Using g As Graphics = Graphics.FromImage(eraserBitmap)
            g.FillRectangle(Brushes.Black, 0, 0, cursorSize, cursorSize)
        End Using
        Dim eraserIcon As IntPtr = eraserBitmap.GetHicon()
        eraserCursor = New Cursor(eraserIcon)
    End Sub

    Private Sub btnOpenFolder_Click(sender As Object, e As EventArgs) Handles btnOpenFolder.Click
        If FolderBrowserDialog1.ShowDialog() = DialogResult.OK Then
            selectedFolderPath = FolderBrowserDialog1.SelectedPath
            images = Directory.GetFiles(selectedFolderPath, "*.*").
                Where(Function(f) Not Path.GetFileName(f).ToLower().Contains("row_") AndAlso
                                  Not Path.GetFileName(f).ToLower().Contains("mask") AndAlso
                                  Not System.Text.RegularExpressions.Regex.IsMatch(Path.GetFileNameWithoutExtension(f), "^[{]?[0-9a-fA-F]{8}-([0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}[}]?$") AndAlso
                                  (f.EndsWith(".jpg") OrElse f.EndsWith(".png") OrElse f.EndsWith(".tiff"))).ToList()
            LoadLabels()
            If images.Count > 0 Then
                currentIndex = 0
                LoadImage()
            Else
                MessageBox.Show("No images found in the selected folder.", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error)
            End If
        End If
    End Sub

    Private Sub LoadLabels()
        labels.Clear()
        Dim csvPath As String = Path.Combine(selectedFolderPath, "labels.csv")
        If File.Exists(csvPath) Then
            Dim lines = File.ReadAllLines(csvPath)
            For Each line In lines.Skip(1) ' Skip header
                Dim parts = line.Split(","c)
                If parts.Length >= 2 Then
                    Dim imageName = parts(0).Trim()
                    Dim label = parts(1).Trim().Trim(""""c) ' Remove extra quotes
                    labels(imageName) = label
                End If
            Next
        End If
    End Sub

    Private Sub btnNext_Click(sender As Object, e As EventArgs) Handles btnNext.Click
        SaveCurrentLabel()
        MoveToNextImage()
    End Sub

    Private Sub btnPrev_Click(sender As Object, e As EventArgs) Handles btnPrev.Click
        SaveCurrentLabel()
        MoveToPreviousImage()
    End Sub

    Private Sub btnSaveLabel_Click(sender As Object, e As EventArgs) Handles btnSaveLabel.Click
        SaveCurrentLabel()
    End Sub

    Private Sub btnDelete_Click(sender As Object, e As EventArgs) Handles btnDelete.Click
        If currentIndex >= 0 AndAlso currentIndex < images.Count Then
            Dim imagePath As String = images(currentIndex)

            ' Remove image from PictureBox and dispose it
            If PictureBox1.Image IsNot Nothing Then
                PictureBox1.Image.Dispose()
                PictureBox1.Image = Nothing
            End If
            If PictureBox2.Image IsNot Nothing Then
                PictureBox2.Image.Dispose()
                PictureBox2.Image = Nothing
            End If
            Application.DoEvents() ' Allow UI to update

            ' Delete the image file
            Try
                File.Delete(imagePath)
            Catch ex As Exception
                MessageBox.Show($"Error deleting file: {ex.Message}", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error)
                Return
            End Try

            ' Remove image from list and update current index
            images.RemoveAt(currentIndex)
            If currentIndex >= images.Count Then
                currentIndex = images.Count - 1
            End If

            ' Load next image or clear display if no images left
            If images.Count > 0 Then
                LoadImage()
            Else
                PictureBox1.Image = Nothing
                PictureBox2.Image = Nothing
                lblImageInfo.Text = "No images available"
                txtLabel.Text = ""
            End If
        End If
    End Sub

    Private Sub btnSkipRight_Click(sender As Object, e As EventArgs) Handles btnSkipRight.Click
        MoveToNextImage()
    End Sub

    Private Sub btnSkipLeft_Click(sender As Object, e As EventArgs) Handles btnSkipLeft.Click
        MoveToPreviousImage()
    End Sub

    Private Sub LoadImage()
        If images.Count > 0 AndAlso currentIndex >= 0 AndAlso currentIndex < images.Count Then
            ' Use a memory stream to ensure the file is not locked
            Using imgStream As New MemoryStream(File.ReadAllBytes(images(currentIndex)))
                Dim img = Image.FromStream(imgStream)
                If Not IsImageWhite(img) Then
                    originalImage = ConvertToNonIndexedBitmap(CType(img.Clone(), Bitmap)) ' Save the original image for erasing
                    maskImage = New Bitmap(originalImage.Width, originalImage.Height, PixelFormat.Format32bppArgb)

                    Dim resizedImage = ResizeImageToFit(img, PictureBox1.Width, PictureBox1.Height)
                    PictureBox1.Image = resizedImage
                    Dim imageName = Path.GetFileName(images(currentIndex))

                    If labels.ContainsKey(imageName) Then
                        txtLabel.Text = labels(imageName)
                    Else
                        txtLabel.Text = "" ' Clear the text box for a new label
                    End If

                    Dim wordFilePattern As String = "seg_(.*?)_(\d{3})_(\d{3})_(\d{3})\.tiff"
                    Dim splitFilePattern As String = "row_(.*?)_(\d{3})\.tiff"

                    ' Extract parts from the word file name
                    Dim wordMatch As Match = Regex.Match(imageName, wordFilePattern)
                    If wordMatch.Success Then
                        Dim baseName As String = wordMatch.Groups(1).Value
                        Dim lineNumber As String = wordMatch.Groups(2).Value

                        ' Construct the corresponding splits file path
                        Dim splitsImagePath As String = Path.Combine(selectedFolderPath, $"row_{baseName}_{lineNumber}.tiff")

                        If File.Exists(splitsImagePath) Then
                            Using splitsStream As New MemoryStream(File.ReadAllBytes(splitsImagePath))
                                Dim splitsImg As Image = Image.FromStream(splitsStream)
                                Dim resizedSplitsImage As Image = ResizeImageToFit(splitsImg, PictureBox2.Width, PictureBox2.Height)
                                PictureBox2.Image = resizedSplitsImage
                            End Using
                        Else
                            lblImageInfo.Text = "Could not find split file for " + splitsImagePath
                        End If
                    Else
                        MoveToNextImage()
                    End If
                End If
            End Using
        End If
    End Sub

    Private Function IsImageWhite(img As Image) As Boolean
        Dim bmp As New Bitmap(img)
        For y As Integer = 0 To bmp.Height - 1
            For x As Integer = 0 To bmp.Width - 1
                If bmp.GetPixel(x, y) <> Color.White Then
                    Return False
                End If
            Next
        Next
        Return True
    End Function

    Private Function ResizeImageToFit(originalImage As Image, targetWidth As Integer, targetHeight As Integer) As Image
        Dim originalWidth As Integer = originalImage.Width
        Dim originalHeight As Integer = originalImage.Height

        Dim ratioX As Double = CDbl(targetWidth) / CDbl(originalWidth)
        Dim ratioY As Double = CDbl(targetHeight) / CDbl(originalHeight)
        Dim ratio As Double = Math.Min(ratioX, ratioY)

        Dim newWidth As Integer = CInt(originalWidth * ratio)
        Dim newHeight As Integer = CInt(originalHeight * ratio)

        Dim resizedImage As New Bitmap(newWidth, newHeight)
        Using graphics As Graphics = Graphics.FromImage(resizedImage)
            graphics.DrawImage(originalImage, 0, 0, newWidth, newHeight)
        End Using

        Return resizedImage
    End Function

    Private Function ConvertToNonIndexedBitmap(source As Bitmap) As Bitmap
        Dim newBmp As New Bitmap(source.Width, source.Height, PixelFormat.Format32bppArgb)
        Using g As Graphics = Graphics.FromImage(newBmp)
            g.DrawImage(source, 0, 0)
        End Using
        Return newBmp
    End Function

    Private Sub SaveCurrentLabel()
        If PictureBox1.Image IsNot Nothing Then
            Dim currentImagePath = images(currentIndex)
            Dim currentImageName = Path.GetFileName(currentImagePath)
            If Not String.IsNullOrWhiteSpace(txtLabel.Text) Then
                ' Generate a unique ID and rename the file
                Dim uniqueID = Guid.NewGuid().ToString()
                Dim newFileName = $"{uniqueID}{Path.GetExtension(currentImagePath)}"
                Dim newFilePath = Path.Combine(selectedFolderPath, newFileName)
                File.Move(currentImagePath, newFilePath)

                ' Move a copy of the original file to the "processed" folder
                Dim processedFolderPath = Path.Combine(selectedFolderPath, "processed")
                If Not Directory.Exists(processedFolderPath) Then
                    Directory.CreateDirectory(processedFolderPath)
                End If
                Dim processedFilePath = Path.Combine(processedFolderPath, Path.GetFileName(currentImagePath))
                File.Copy(newFilePath, processedFilePath, overwrite:=True)

                ' Update the images list and dictionary
                images(currentIndex) = newFilePath
                If labels.ContainsKey(currentImageName) Then
                    labels.Remove(currentImageName)
                End If
                labels(newFileName) = txtLabel.Text

                ' Save the label and mask to the CSV file
                SaveLabelAndMask(newFilePath, txtLabel.Text)
                txtLabel.Text = "" ' Clear the text box after saving
            ElseIf labels.ContainsKey(currentImageName) Then
                ' Remove the label from the CSV if the text box is empty
                labels.Remove(currentImageName)
                RemoveLabelFromCSV(currentImageName)
            End If
        End If
    End Sub

    Private Sub SaveLabelAndMask(imagePath As String, label As String)
        Dim csvPath As String = Path.Combine(selectedFolderPath, "labels.csv")
        Dim maskFileName As String = $"{Path.GetFileNameWithoutExtension(imagePath)}_mask.png"
        Dim maskFilePath As String = Path.Combine(selectedFolderPath, maskFileName)

        ' Save the mask image
        If maskImage IsNot Nothing Then
            ' Convert the image displayed in PictureBox1 to a bitmap
            Dim displayedImage As Bitmap = ConvertToNonIndexedBitmap(CType(PictureBox1.Image, Bitmap))

            ' Resize the displayed image back to the original dimensions
            Dim fullSizeMask As New Bitmap(originalImage.Width, originalImage.Height)
            Using g As Graphics = Graphics.FromImage(fullSizeMask)
                g.DrawImage(displayedImage, 0, 0, originalImage.Width, originalImage.Height)
            End Using

            ' Convert the resized image to binary
            For y As Integer = 0 To fullSizeMask.Height - 1
                For x As Integer = 0 To fullSizeMask.Width - 1
                    Dim pixelColor As Color = fullSizeMask.GetPixel(x, y)
                    If pixelColor.R < 128 AndAlso pixelColor.G < 128 AndAlso pixelColor.B < 128 Then
                        fullSizeMask.SetPixel(x, y, Color.Black)
                    Else
                        fullSizeMask.SetPixel(x, y, Color.White)
                    End If
                Next
            Next

            fullSizeMask.Save(maskFilePath, ImageFormat.Png)
        End If

        ' Save label and mask file info to CSV
        Dim csvLine As String = $"{Path.GetFileName(imagePath)}, ""{label}"", {maskFileName}"

        If Not File.Exists(csvPath) Then
            File.WriteAllText(csvPath, "Image,Label,Mask" & Environment.NewLine)
        End If

        File.AppendAllText(csvPath, csvLine & Environment.NewLine)
    End Sub

    Private Sub RemoveLabelFromCSV(imageName As String)
        Dim csvPath As String = Path.Combine(selectedFolderPath, "labels.csv")
        If File.Exists(csvPath) Then
            Dim lines = File.ReadAllLines(csvPath).ToList()
            Dim newLines = lines.Where(Function(line) Not line.StartsWith(imageName)).ToList()
            File.WriteAllLines(csvPath, newLines)
        End If
    End Sub

    Private Sub MoveToNextImage()
        If currentIndex < images.Count - 1 Then
            currentIndex += 1
        Else
            currentIndex = 0
        End If
        LoadImage()
    End Sub

    Private Sub MoveToPreviousImage()
        If currentIndex > 0 Then
            currentIndex -= 1
        Else
            currentIndex = images.Count - 1
        End If
        LoadImage()
    End Sub

    Private Sub Form1_KeyDown(sender As Object, e As KeyEventArgs) Handles Me.KeyDown
        If e.KeyCode = Keys.Tab Then
            e.SuppressKeyPress = True
            If e.Shift Then
                btnPrev.PerformClick()
            Else
                btnNext.PerformClick()
            End If
        End If
    End Sub

    Private Sub btnPathImage_Click(sender As Object, e As EventArgs) Handles btnPathImage.Click
        ' Configure the OpenFileDialog
        OpenFileDialog1.Filter = "Image Files|*.bmp;*.jpg;*.jpeg;*.png;*.gif|All Files|*.*"
        OpenFileDialog1.Title = "Select an Image"

        ' Show the dialog and get the result
        If OpenFileDialog1.ShowDialog() = DialogResult.OK Then
            ' Get the selected file path
            Dim imagePath As String = OpenFileDialog1.FileName

            ' Load the image
            Dim image As Image = Image.FromFile(imagePath)

            ' Create an instance of SegmentedImage form
            Dim segmentedImageForm As New SegmentedImage

            ' Pass the image to the SegmentedImage form
            segmentedImageForm.SetImage(image)

            ' Show the SegmentedImage form
            segmentedImageForm.Show()
        End If
    End Sub

    Private Sub PictureBox1_MouseEnter(sender As Object, e As EventArgs) Handles PictureBox1.MouseEnter
        PictureBox1.Cursor = eraserCursor
    End Sub

    Private Sub PictureBox1_MouseLeave(sender As Object, e As EventArgs) Handles PictureBox1.MouseLeave
        PictureBox1.Cursor = Cursors.Default
    End Sub

    Private Sub PictureBox1_MouseDown(sender As Object, e As MouseEventArgs) Handles PictureBox1.MouseDown
        If e.Button = MouseButtons.Left Then
            isErasing = True
            ErasePixels(e.Location)
        End If
    End Sub

    Private Sub PictureBox1_MouseMove(sender As Object, e As MouseEventArgs) Handles PictureBox1.MouseMove
        If isErasing Then
            ErasePixels(e.Location)
        End If
    End Sub

    Private Sub PictureBox1_MouseUp(sender As Object, e As MouseEventArgs) Handles PictureBox1.MouseUp
        If e.Button = MouseButtons.Left Then
            isErasing = False
        End If
    End Sub

    Private Sub ErasePixels(location As Point)
        If originalImage Is Nothing Then Return

        ' Calculate actual position in the original image
        Dim pictureBoxAspectRatio As Double = CDbl(PictureBox1.Width) / PictureBox1.Height
        Dim imageAspectRatio As Double = CDbl(originalImage.Width) / originalImage.Height

        Dim actualX, actualY As Integer

        If pictureBoxAspectRatio > imageAspectRatio Then
            ' PictureBox is wider than the image, calculate based on height
            Dim scaleFactor As Double = CDbl(originalImage.Height) / PictureBox1.Height
            actualY = CInt(location.Y * scaleFactor)
            Dim widthOffset As Integer = CInt((PictureBox1.Width - (originalImage.Width / scaleFactor)) / 2)
            actualX = CInt((location.X - widthOffset) * scaleFactor)
        Else
            ' PictureBox is taller than the image, calculate based on width
            Dim scaleFactor As Double = CDbl(originalImage.Width) / PictureBox1.Width
            actualX = CInt(location.X * scaleFactor)
            Dim heightOffset As Integer = CInt((PictureBox1.Height - (originalImage.Height / scaleFactor)) / 2)
            actualY = CInt((location.Y - heightOffset) * scaleFactor)
        End If

        ' Ensure coordinates are within bounds
        If actualX < 0 OrElse actualX >= originalImage.Width OrElse actualY < 0 OrElse actualY >= originalImage.Height Then Return

        ' Perform the erase operation on the original image
        Using g As Graphics = Graphics.FromImage(originalImage)
            g.FillRectangle(Brushes.White, actualX, actualY, eraserSize, eraserSize)
        End Using

        ' Perform the erase operation on the mask image
        Using g As Graphics = Graphics.FromImage(maskImage)
            g.FillRectangle(Brushes.Black, actualX, actualY, eraserSize, eraserSize)
        End Using

        ' Update the PictureBox1 image with the modified original image
        PictureBox1.Image = ResizeImageToFit(originalImage, PictureBox1.Width, PictureBox1.Height)
    End Sub
End Class
