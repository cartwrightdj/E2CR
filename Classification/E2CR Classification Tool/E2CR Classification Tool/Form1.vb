Imports System.IO
Imports System.Net.Mime.MediaTypeNames
Imports System.Text.RegularExpressions
Imports System.Drawing
Imports System.Windows.Forms.VisualStyles.VisualStyleElement

Public Class Form1
    Private imagePaths As List(Of String)
    Private currentIndex As Integer = -1
    Private classifications As Dictionary(Of String, String)
    Private images As List(Of String)
    Private selectedFolderPath As String


    Private Sub Form1_Load(sender As Object, e As EventArgs) Handles MyBase.Load
        imagePaths = New List(Of String)
        classifications = New Dictionary(Of String, String)
        ' Setup PictureBox size mode and initial properties
        pbImagesToLabel.SizeMode = PictureBoxSizeMode.Zoom
        pbImagesToLabel.Dock = DockStyle.Fill
        Panel1.AutoScroll = True
        Panel1.Controls.Add(pbImagesToLabel)
        pbRow.SizeMode = PictureBoxSizeMode.Zoom

        ' Setup VScrollBar
        HScrollBar1.Minimum = 0
        HScrollBar1.Visible = False

        AddHandler HScrollBar1.Scroll, AddressOf hScrollBar1_Scroll
    End Sub

    Private Sub btnSelectFolder_Click(sender As Object, e As EventArgs) Handles btnSelectFolder.Click
        Using folderBrowserDialog As New FolderBrowserDialog()
            If folderBrowserDialog.ShowDialog() = DialogResult.OK Then
                selectedFolderPath = folderBrowserDialog.SelectedPath

                images = Directory.GetFiles(selectedFolderPath, "*.*").
                    Where(Function(f) Not Path.GetFileName(f).ToLower().Contains("xxx_") AndAlso
                                      Not Path.GetFileName(f).ToLower().Contains("mask") AndAlso
                                      Not Regex.IsMatch(Path.GetFileNameWithoutExtension(f), "^[{]?[0-9a-fA-F]{8}-([0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}[}]?$") AndAlso
                                      (f.EndsWith(".jpg") OrElse f.EndsWith(".png") OrElse f.EndsWith(".tiff"))).ToList()


                If images.Count > 0 Then
                    txtLabel.Enabled = True
                    currentIndex = 0
                    HScrollBar1.Maximum = images.Count - 1
                    HScrollBar1.Visible = True
                    LoadImage()
                Else
                    MessageBox.Show("No images found in the selected folder.", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error)
                End If
            End If
        End Using
    End Sub

    Private Sub LoadImage()
        If images IsNot Nothing AndAlso images.Count > 0 Then
            Dim fileName As String = Path.GetFileName(images(currentIndex))


            txtImageName.Text = fileName
            Dim img = System.Drawing.Image.FromFile(images(currentIndex))
            pbImagesToLabel.Image = img
            pbImagesToLabel.Width = img.Width
            pbImagesToLabel.Height = img.Height

            Dim rowPattern As String = "contour_(.*?)_(\d{3})_(\d{3})_(\d{3}).tiff"
            Dim wordMatch As Match = Regex.Match(fileName, rowPattern)
            If wordMatch.Success Then
                Dim baseName As String = wordMatch.Groups(1).Value
                Dim lineNumber As String = wordMatch.Groups(2).Value

                ' Construct the corresponding splits file path
                Dim splitsImagePath As String = Path.Combine(selectedFolderPath, $"row_{baseName}_{lineNumber}.tiff")
                Dim rowcsPath As String = Path.Combine(selectedFolderPath, $"rowcs_{baseName}_{lineNumber}.tiff")
                If File.Exists(splitsImagePath) Then
                    pbRow.Image = System.Drawing.Image.FromFile(splitsImagePath)
                    pbRowCS.Image = System.Drawing.Image.FromFile(rowcsPath)
                Else
                    'lblImageInfo.Text = "Could not find split file for " + splitsImagePath
                End If
            End If

        End If
    End Sub

    Private Sub hScrollBar1_Scroll(sender As Object, e As ScrollEventArgs)
        currentIndex = HScrollBar1.Value
        LoadImage()
    End Sub
End Class