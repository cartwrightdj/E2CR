<Global.Microsoft.VisualBasic.CompilerServices.DesignerGenerated()>
Partial Class labeling_main
    Inherits System.Windows.Forms.Form

    'Form overrides dispose to clean up the component list.
    <System.Diagnostics.DebuggerNonUserCode()>
    Protected Overrides Sub Dispose(ByVal disposing As Boolean)
        Try
            If disposing AndAlso components IsNot Nothing Then
                components.Dispose()
            End If
        Finally
            MyBase.Dispose(disposing)
        End Try
    End Sub

    'Required by the Windows Form Designer
    Private components As System.ComponentModel.IContainer

    'NOTE: The following procedure is required by the Windows Form Designer
    'It can be modified using the Windows Form Designer.  
    'Do not modify it using the code editor.
    <System.Diagnostics.DebuggerStepThrough()>
    Private Sub InitializeComponent()
        Dim resources As System.ComponentModel.ComponentResourceManager = New System.ComponentModel.ComponentResourceManager(GetType(labeling_main))
        PictureToLabel_box = New PictureBox()
        HeightLabel = New Label()
        WidthLabel = New Label()
        BreakPoints = New TextBox()
        ClearLinesButton = New Button()
        ExportButton = New Button()
        NextImg = New Button()
        PrevImg = New Button()
        FileNameLabel = New Label()
        ToolStrip1 = New ToolStrip()
        LoadImageToolButton = New ToolStripButton()
        SaveImageToolButton = New ToolStripButton()
        StatusStrip1 = New StatusStrip()
        StatusLabel = New ToolStripStatusLabel()
        CType(PictureToLabel_box, ComponentModel.ISupportInitialize).BeginInit()
        ToolStrip1.SuspendLayout()
        StatusStrip1.SuspendLayout()
        SuspendLayout()
        ' 
        ' PictureToLabel_box
        ' 
        PictureToLabel_box.BorderStyle = BorderStyle.Fixed3D
        PictureToLabel_box.Location = New Point(109, 93)
        PictureToLabel_box.Name = "PictureToLabel_box"
        PictureToLabel_box.Size = New Size(1254, 180)
        PictureToLabel_box.TabIndex = 0
        PictureToLabel_box.TabStop = False
        ' 
        ' HeightLabel
        ' 
        HeightLabel.AutoSize = True
        HeightLabel.Location = New Point(0, 154)
        HeightLabel.Name = "HeightLabel"
        HeightLabel.Size = New Size(13, 15)
        HeightLabel.TabIndex = 2
        HeightLabel.Text = "0"
        ' 
        ' WidthLabel
        ' 
        WidthLabel.AutoSize = True
        WidthLabel.Location = New Point(611, 75)
        WidthLabel.Name = "WidthLabel"
        WidthLabel.Size = New Size(13, 15)
        WidthLabel.TabIndex = 3
        WidthLabel.Text = "0"
        ' 
        ' BreakPoints
        ' 
        BreakPoints.Location = New Point(200, 280)
        BreakPoints.Name = "BreakPoints"
        BreakPoints.Size = New Size(981, 23)
        BreakPoints.TabIndex = 4
        ' 
        ' ClearLinesButton
        ' 
        ClearLinesButton.Location = New Point(1187, 279)
        ClearLinesButton.Name = "ClearLinesButton"
        ClearLinesButton.Size = New Size(85, 23)
        ClearLinesButton.TabIndex = 5
        ClearLinesButton.Text = "Clear Lines"
        ClearLinesButton.UseVisualStyleBackColor = True
        ' 
        ' ExportButton
        ' 
        ExportButton.Location = New Point(1278, 279)
        ExportButton.Name = "ExportButton"
        ExportButton.Size = New Size(85, 23)
        ExportButton.TabIndex = 6
        ExportButton.Text = "Export"
        ExportButton.UseVisualStyleBackColor = True
        ' 
        ' NextImg
        ' 
        NextImg.Location = New Point(1392, 152)
        NextImg.Name = "NextImg"
        NextImg.Size = New Size(49, 54)
        NextImg.TabIndex = 7
        NextImg.Text = "Button1"
        NextImg.UseVisualStyleBackColor = True
        ' 
        ' PrevImg
        ' 
        PrevImg.Location = New Point(40, 154)
        PrevImg.Name = "PrevImg"
        PrevImg.Size = New Size(49, 54)
        PrevImg.TabIndex = 8
        PrevImg.Text = "Button1"
        PrevImg.UseVisualStyleBackColor = True
        ' 
        ' FileNameLabel
        ' 
        FileNameLabel.AutoSize = True
        FileNameLabel.BackColor = SystemColors.GradientActiveCaption
        FileNameLabel.Location = New Point(109, 306)
        FileNameLabel.Name = "FileNameLabel"
        FileNameLabel.Size = New Size(41, 15)
        FileNameLabel.TabIndex = 9
        FileNameLabel.Text = "Label1"
        ' 
        ' ToolStrip1
        ' 
        ToolStrip1.Items.AddRange(New ToolStripItem() {LoadImageToolButton, SaveImageToolButton})
        ToolStrip1.Location = New Point(0, 0)
        ToolStrip1.Name = "ToolStrip1"
        ToolStrip1.Size = New Size(1500, 25)
        ToolStrip1.TabIndex = 10
        ToolStrip1.Text = "ToolStrip1"
        ' 
        ' LoadImageToolButton
        ' 
        LoadImageToolButton.DisplayStyle = ToolStripItemDisplayStyle.Image
        LoadImageToolButton.Image = CType(resources.GetObject("LoadImageToolButton.Image"), Image)
        LoadImageToolButton.ImageTransparentColor = Color.Magenta
        LoadImageToolButton.Name = "LoadImageToolButton"
        LoadImageToolButton.Size = New Size(23, 22)
        LoadImageToolButton.Text = "ToolStripButton2"
        ' 
        ' SaveImageToolButton
        ' 
        SaveImageToolButton.DisplayStyle = ToolStripItemDisplayStyle.Image
        SaveImageToolButton.Image = CType(resources.GetObject("SaveImageToolButton.Image"), Image)
        SaveImageToolButton.ImageTransparentColor = Color.Magenta
        SaveImageToolButton.Name = "SaveImageToolButton"
        SaveImageToolButton.Size = New Size(23, 22)
        SaveImageToolButton.Text = "SaveImageToolButton"
        ' 
        ' StatusStrip1
        ' 
        StatusStrip1.Items.AddRange(New ToolStripItem() {StatusLabel})
        StatusStrip1.Location = New Point(0, 374)
        StatusStrip1.Name = "StatusStrip1"
        StatusStrip1.Size = New Size(1500, 22)
        StatusStrip1.TabIndex = 11
        StatusStrip1.Text = "StatusStrip1"
        ' 
        ' StatusLabel
        ' 
        StatusLabel.Name = "StatusLabel"
        StatusLabel.Size = New Size(119, 17)
        StatusLabel.Text = "ToolStripStatusLabel1"
        ' 
        ' labeling_main
        ' 
        AutoScaleDimensions = New SizeF(7F, 15F)
        AutoScaleMode = AutoScaleMode.Font
        BackColor = SystemColors.GradientActiveCaption
        ClientSize = New Size(1500, 396)
        Controls.Add(StatusStrip1)
        Controls.Add(ToolStrip1)
        Controls.Add(FileNameLabel)
        Controls.Add(PrevImg)
        Controls.Add(NextImg)
        Controls.Add(ExportButton)
        Controls.Add(ClearLinesButton)
        Controls.Add(BreakPoints)
        Controls.Add(WidthLabel)
        Controls.Add(HeightLabel)
        Controls.Add(PictureToLabel_box)
        Name = "labeling_main"
        Text = "E2CR Row Break Labeling Tool"
        CType(PictureToLabel_box, ComponentModel.ISupportInitialize).EndInit()
        ToolStrip1.ResumeLayout(False)
        ToolStrip1.PerformLayout()
        StatusStrip1.ResumeLayout(False)
        StatusStrip1.PerformLayout()
        ResumeLayout(False)
        PerformLayout()
    End Sub

    Friend WithEvents PictureToLabel_box As PictureBox
    Friend WithEvents LoadImageButton As Button
    Friend WithEvents HeightLabel As Label
    Friend WithEvents WidthLabel As Label
    Friend WithEvents BreakPoints As TextBox
    Friend WithEvents ClearLinesButton As Button
    Friend WithEvents ExportButton As Button
    Friend WithEvents NextImg As Button
    Friend WithEvents PrevImg As Button
    Friend WithEvents FileNameLabel As Label
    Friend WithEvents ToolStrip1 As ToolStrip
    Friend WithEvents SaveImageToolButton As ToolStripButton
    Friend WithEvents StatusStrip1 As StatusStrip
    Friend WithEvents StatusLabel As ToolStripStatusLabel
    Friend WithEvents LoadImageToolButton As ToolStripButton

End Class
