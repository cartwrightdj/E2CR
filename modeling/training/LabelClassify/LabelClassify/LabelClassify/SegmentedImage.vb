Public Class SegmentedImage
    Private zoomFactor As Double = 1.0
    Private isDragging As Boolean = False
    Private startPoint As Point

    ' This method sets the image in PictureBox and adjusts the Panel and Form size
    Public Sub SetImage(image As Image)
        ' Set the PictureBox to display the image
        PictureBox1.Image = image

        ' Adjust the PictureBox size mode to auto-size
        PictureBox1.SizeMode = PictureBoxSizeMode.AutoSize

        ' Adjust the Panel size to fit the PictureBox
        Panel1.AutoScroll = True
        Panel1.Controls.Add(PictureBox1)
        PictureBox1.Location = New Point(0, 0)

        ' Adjust the Form size to fit the Panel
        Me.ClientSize = New Size(Panel1.Width + 20, Panel1.Height + 20)

        ' If the image is larger than the Panel, the scroll bars will appear automatically
        If PictureBox1.Width > Panel1.ClientSize.Width Or PictureBox1.Height > Panel1.ClientSize.Height Then
            Panel1.AutoScrollMinSize = New Size(PictureBox1.Width, PictureBox1.Height)
        Else
            Panel1.AutoScrollMinSize = New Size(0, 0)
        End If

        ' Add mouse event handlers for dragging and zooming
        AddHandler PictureBox1.MouseDown, AddressOf PictureBox1_MouseDown
        AddHandler PictureBox1.MouseMove, AddressOf PictureBox1_MouseMove
        AddHandler PictureBox1.MouseUp, AddressOf PictureBox1_MouseUp
        AddHandler PictureBox1.MouseWheel, AddressOf PictureBox1_MouseWheel
        PictureBox1.Focus() ' Ensure PictureBox1 can capture mouse wheel events
    End Sub

    Private Sub PictureBox1_MouseDown(sender As Object, e As MouseEventArgs)
        If e.Button = MouseButtons.Left Then
            isDragging = True
            startPoint = e.Location
            PictureBox1.Cursor = Cursors.Hand
        End If
    End Sub

    Private Sub PictureBox1_MouseMove(sender As Object, e As MouseEventArgs)
        If isDragging Then
            Dim endPoint As Point = e.Location
            Dim offset As Point = New Point(endPoint.X - startPoint.X, endPoint.Y - startPoint.Y)
            Panel1.AutoScrollPosition = New Point(-Panel1.AutoScrollPosition.X - offset.X, -Panel1.AutoScrollPosition.Y - offset.Y)
        End If
    End Sub

    Private Sub PictureBox1_MouseUp(sender As Object, e As MouseEventArgs)
        If e.Button = MouseButtons.Left Then
            isDragging = False
            PictureBox1.Cursor = Cursors.Default
        End If
    End Sub

    Private Sub PictureBox1_MouseWheel(sender As Object, e As MouseEventArgs)
        If Control.ModifierKeys = Keys.Control Then
            Dim zoomChange As Double = 0.1
            If e.Delta > 0 Then
                zoomFactor += zoomChange
            ElseIf e.Delta < 0 Then
                zoomFactor -= zoomChange
            End If

            If zoomFactor < 0.1 Then
                zoomFactor = 0.1
            End If

            PictureBox1.Size = New Size(CInt(PictureBox1.Image.Width * zoomFactor), CInt(PictureBox1.Image.Height * zoomFactor))
            Panel1.AutoScrollMinSize = New Size(PictureBox1.Width, PictureBox1.Height)
        End If
    End Sub

    Private Sub SegmentedImage_Resize(sender As Object, e As EventArgs) Handles MyBase.Resize
        ' Resize the Panel to fit the form
        Panel1.Size = New Size(Me.ClientSize.Width - 20, Me.ClientSize.Height - 20)

        ' Adjust the PictureBox size
        If PictureBox1.Image IsNot Nothing Then
            If PictureBox1.Width > Panel1.ClientSize.Width Or PictureBox1.Height > Panel1.ClientSize.Height Then
                Panel1.AutoScrollMinSize = New Size(PictureBox1.Width, PictureBox1.Height)
            Else
                Panel1.AutoScrollMinSize = New Size(0, 0)
            End If
        End If
    End Sub
End Class
