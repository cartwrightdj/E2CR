Imports System.Drawing
Imports System.Drawing.Imaging
Imports System.IO

Public Class labeling_main
    Private originalImage As Bitmap
    Private imageLoaded As Boolean = False
    Private lines As New List(Of Point) ' To keep track of drawn lines and their coordinates
    Private previousBreakPointsText As String ' To keep track of the previous BreakPoints text
    Private originalImageFileName As String ' To keep track of the original image file name with extension
    Private originalImageFilePath As String ' To keep track of the original image file path
    Private imageFiles As List(Of String) ' To keep track of all image files in the directory
    Private currentIndex As Integer ' To keep track of the current image index

    Private Sub labeling_main_Load(sender As Object, e As EventArgs) Handles MyBase.Load
        ' Set the PictureBox SizeMode to adjust the image to fit the PictureBox
        PictureToLabel_box.SizeMode = PictureBoxSizeMode.Normal
    End Sub



    Private Sub LoadImage(filePath As String)
        Try
            ' Load the selected image into the PictureBox
            Dim loadedImage As Image = Image.FromFile(filePath)
            ' Convert the image to a non-indexed pixel format
            originalImage = ConvertToNonIndexed(loadedImage)
            PictureToLabel_box.Image = DirectCast(originalImage.Clone(), Image)
            imageLoaded = True ' Set the flag to indicate the image has been loaded

            ' Store the original image file name with extension
            originalImageFileName = Path.GetFileName(filePath)
            ' Store the original image file path
            originalImageFilePath = filePath

            ' Display the image dimensions in the labels
            HeightLabel.Text = "Height: " & originalImage.Height.ToString()
            WidthLabel.Text = "Width: " & originalImage.Width.ToString()

            ' Adjust PictureBox size to fit the image if the image is smaller
            If originalImage.Width <= PictureToLabel_box.Width AndAlso originalImage.Height <= PictureToLabel_box.Height Then
                PictureToLabel_box.SizeMode = PictureBoxSizeMode.CenterImage
            Else
                PictureToLabel_box.SizeMode = PictureBoxSizeMode.StretchImage
            End If

            ' Clear previous lines and BreakPoints
            lines.Clear()
            BreakPoints.Clear()
            previousBreakPointsText = String.Empty
        Catch ex As Exception
            MessageBox.Show("Error loading image: " & ex.Message)
        End Try
    End Sub

    Private Function ConvertToNonIndexed(image As Image) As Bitmap
        Dim bmp As New Bitmap(image.Width, image.Height, PixelFormat.Format32bppArgb)
        Using gr As Graphics = Graphics.FromImage(bmp)
            gr.DrawImage(image, New Rectangle(0, 0, bmp.Width, bmp.Height))
        End Using
        Return bmp
    End Function

    Private Sub PictureToLabel_box_MouseClick(sender As Object, e As MouseEventArgs) Handles PictureToLabel_box.MouseClick
        If imageLoaded AndAlso PictureToLabel_box.Image IsNot Nothing Then
            Dim clickX As Integer = e.X
            Dim clickY As Integer = e.Y

            If PictureToLabel_box.SizeMode = PictureBoxSizeMode.CenterImage Then
                ' Calculate the offset if the image is centered
                Dim offsetX As Integer = (PictureToLabel_box.Width - originalImage.Width) / 2
                Dim offsetY As Integer = (PictureToLabel_box.Height - originalImage.Height) / 2

                clickX -= offsetX
                clickY -= offsetY
            ElseIf PictureToLabel_box.SizeMode = PictureBoxSizeMode.StretchImage Then
                ' Calculate the scaling factors
                Dim scaleX As Double = CDbl(originalImage.Width) / PictureToLabel_box.Width
                Dim scaleY As Double = CDbl(originalImage.Height) / PictureToLabel_box.Height

                ' Convert the click coordinates to the image coordinates
                clickX = CInt(e.X * scaleX)
                clickY = CInt(e.Y * scaleY)
            End If

            ' Ensure the click coordinates are within the image bounds
            If clickX < 0 OrElse clickY < 0 OrElse clickX >= originalImage.Width OrElse clickY >= originalImage.Height Then
                Return
            End If

            ' Find the nearest line within a threshold
            Dim threshold As Integer = 5
            Dim existingPoint As Point? = Nothing
            For Each point In lines
                If Math.Abs(point.X - clickX) < threshold AndAlso Math.Abs(point.Y - clickY) < threshold Then
                    existingPoint = point
                    Exit For
                End If
            Next

            If existingPoint IsNot Nothing Then
                ' Remove the line and the entry in BreakPoints
                lines.Remove(existingPoint.Value)
                UpdateImage()
                UpdateBreakPoints()
            Else
                ' Add the line to the list
                lines.Add(New Point(clickX, clickY))
                ' Draw the new line
                DrawLines()
                ' Update the BreakPoints TextBox
                UpdateBreakPoints()
            End If
        End If
    End Sub

    Private Sub DrawLines()
        Dim updatedImage As Bitmap = DirectCast(originalImage.Clone(), Bitmap)
        Using g As Graphics = Graphics.FromImage(updatedImage)
            Dim pen As New Pen(Color.Red, 2)
            For Each point In lines
                g.DrawLine(pen, New Point(point.X, 0), New Point(point.X, originalImage.Height))
            Next
        End Using
        PictureToLabel_box.Image = updatedImage
        PictureToLabel_box.Refresh()
    End Sub

    Private Sub UpdateImage()
        ' Redraw the original image without the specific line
        PictureToLabel_box.Image = DirectCast(originalImage.Clone(), Image)
        DrawLines()
    End Sub

    Private Sub UpdateBreakPoints()
        ' Sort the x-coordinates
        Dim sortedX = lines.Select(Function(p) p.X).OrderBy(Function(x) x).ToList()
        ' Update the BreakPoints TextBox with comma-separated values
        BreakPoints.Text = String.Join(", ", sortedX)
        previousBreakPointsText = BreakPoints.Text ' Update the previous text
    End Sub

    Private Sub ClearLinesButton_Click(sender As Object, e As EventArgs) Handles ClearLinesButton.Click
        ' Clear all lines and update the image and BreakPoints TextBox
        lines.Clear()
        UpdateImage()
        BreakPoints.Clear()
    End Sub

    Private Sub BreakPoints_TextChanged(sender As Object, e As EventArgs) Handles BreakPoints.TextChanged
        If Not imageLoaded Then Return

        If String.IsNullOrWhiteSpace(BreakPoints.Text) Then
            lines.Clear()
            UpdateImage()
            previousBreakPointsText = BreakPoints.Text
            Return
        End If

        Try
            ' Parse the new x values from the BreakPoints TextBox
            Dim newXValues = BreakPoints.Text.Split(","c).Select(Function(x) Integer.Parse(x.Trim())).ToList()

            ' Clear the current lines and update with new x values
            lines.Clear()
            For Each x In newXValues
                lines.Add(New Point(x, 0)) ' We assume y=0 for simplicity
            Next

            ' Redraw the lines with the new values
            UpdateImage()
            previousBreakPointsText = BreakPoints.Text ' Update the previous text
        Catch ex As Exception
            ' If an error occurs, revert to the previous text
            MessageBox.Show("Invalid input. Reverting to previous values.")
            BreakPoints.Text = previousBreakPointsText
        End Try
    End Sub


    Private Sub NextImg_Click(sender As Object, e As EventArgs) Handles NextImg.Click
        If Not imageLoaded OrElse imageFiles Is Nothing OrElse currentIndex >= imageFiles.Count - 1 Then
            MessageBox.Show("No next image available.")
            Return
        End If

        currentIndex += 1
        LoadImage(imageFiles(currentIndex))
    End Sub

    Private Sub PrevImg_Click(sender As Object, e As EventArgs) Handles PrevImg.Click
        If Not imageLoaded OrElse imageFiles Is Nothing OrElse currentIndex <= 0 Then
            MessageBox.Show("No previous image available.")
            Return
        End If

        currentIndex -= 1
        LoadImage(imageFiles(currentIndex))
    End Sub

    Private Sub LoadImageToolButton_Click_1(sender As Object, e As EventArgs) Handles LoadImageToolButton.Click
        ' Open a file dialog to select an image file
        Dim openFileDialog As New OpenFileDialog
        openFileDialog.Filter = "Image Files|*.jpg;*.jpeg;*.png;*.bmp;*.gif"

        If openFileDialog.ShowDialog = DialogResult.OK Then
            Try
                ' Get the directory of the selected image
                Dim directory = Path.GetDirectoryName(openFileDialog.FileName)
                ' Get all image files in the directory, ignoring files with "_wlines" in their names
                imageFiles = IO.Directory.GetFiles(directory, "*.*", SearchOption.TopDirectoryOnly).Where(Function(f) ".jpg.jpeg.png.bmp.gif".Contains(Path.GetExtension(f).ToLower) AndAlso Not Path.GetFileNameWithoutExtension(f).Contains("_wlines")).ToList
                ' Get the index of the selected image
                currentIndex = imageFiles.IndexOf(openFileDialog.FileName)

                ' Load the selected image
                LoadImage(imageFiles(currentIndex))
            Catch ex As Exception
                MessageBox.Show("Error loading image: " & ex.Message)
            End Try
        End If
    End Sub

    Private Sub ToolStripButton1_Click(sender As Object, e As EventArgs) Handles SaveImageToolButton.Click
        If Not imageLoaded Then
            MessageBox.Show("No image loaded to export.")
            Return
        End If

        Try
            ' Create the image with lines drawn
            Dim imageWithLines As Bitmap = DirectCast(originalImage.Clone(), Bitmap)
            Using g As Graphics = Graphics.FromImage(imageWithLines)
                Dim pen As New Pen(Color.Red, 2)
                For Each point In lines
                    g.DrawLine(pen, New Point(point.X, 0), New Point(point.X, originalImage.Height))
                Next
            End Using

            ' Save the image with "_wlines" appended to the original filename
            Dim imagePath As String = Path.Combine(Path.GetDirectoryName(originalImageFilePath), Path.GetFileNameWithoutExtension(originalImageFileName) & "_wlines.png")
            imageWithLines.Save(imagePath, ImageFormat.Png)
            StatusLabel.Text = "Image saved successfully: " & imagePath

            ' Save the x values to a CSV file "break_points.csv"
            Dim csvPath As String = Path.Combine(Path.GetDirectoryName(originalImageFilePath), "break_points.csv")
            Dim fileExists As Boolean = File.Exists(csvPath)

            Using writer As New StreamWriter(csvPath, append:=True)
                ' Write headers if the file doesn't exist
                If Not fileExists Then
                    writer.WriteLine("file_name,break_points")
                End If
                writer.WriteLine($"{originalImageFileName},{BreakPoints.Text}")
            End Using
            StatusLabel.Text = "X values saved successfully: " & csvPath

        Catch ex As Exception
            MessageBox.Show("Error saving files: " & ex.Message)
        End Try
    End Sub
End Class
