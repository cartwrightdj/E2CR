Imports System.IO

Public Class Form1
    Private imagePaths As List(Of String)
    Private currentIndex As Integer = -1
    Private classifications As Dictionary(Of String, String)

    Private Sub Form1_Load(sender As Object, e As EventArgs) Handles MyBase.Load
        imagePaths = New List(Of String)
        classifications = New Dictionary(Of String, String)
    End Sub

    Private Sub btnLoadImages_Click(sender As Object, e As EventArgs) Handles btnLoadImages.Click
        Using ofd As New OpenFileDialog()
            ofd.Multiselect = True
            If ofd.ShowDialog() = DialogResult.OK Then
                imagePaths.AddRange(ofd.FileNames)
                DisplayNextImage()
            End If
        End Using
    End Sub

    Private Sub DisplayNextImage()
        currentIndex += 1
        If currentIndex < imagePaths.Count Then
            PictureBox1.Image = Image.FromFile(imagePaths(currentIndex))
        Else
            MessageBox.Show("No more images.")
        End If
    End Sub
End Class