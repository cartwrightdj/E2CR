<Global.Microsoft.VisualBasic.CompilerServices.DesignerGenerated()> _
Partial Class Form1
    Inherits System.Windows.Forms.Form

    'Form overrides dispose to clean up the component list.
    <System.Diagnostics.DebuggerNonUserCode()> _
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
    <System.Diagnostics.DebuggerStepThrough()> _
    Private Sub InitializeComponent()
        Dim resources As System.ComponentModel.ComponentResourceManager = New System.ComponentModel.ComponentResourceManager(GetType(Form1))
        Me.PictureBox1 = New System.Windows.Forms.PictureBox()
        Me.btnSaveLabel = New System.Windows.Forms.Button()
        Me.btnOpenFolder = New System.Windows.Forms.Button()
        Me.txtLabel = New System.Windows.Forms.TextBox()
        Me.btnPrev = New System.Windows.Forms.Button()
        Me.btnNext = New System.Windows.Forms.Button()
        Me.FolderBrowserDialog1 = New System.Windows.Forms.FolderBrowserDialog()
        Me.lblImageInfo = New System.Windows.Forms.Label()
        Me.btnDelete = New System.Windows.Forms.Button()
        Me.Label1 = New System.Windows.Forms.Label()
        Me.PictureBox2 = New System.Windows.Forms.PictureBox()
        Me.btnSkipLeft = New System.Windows.Forms.Button()
        Me.btnSkipRight = New System.Windows.Forms.Button()
        Me.btnPathImage = New System.Windows.Forms.Button()
        Me.OpenFileDialog1 = New System.Windows.Forms.OpenFileDialog()
        CType(Me.PictureBox1, System.ComponentModel.ISupportInitialize).BeginInit()
        CType(Me.PictureBox2, System.ComponentModel.ISupportInitialize).BeginInit()
        Me.SuspendLayout()
        '
        'PictureBox1
        '
        Me.PictureBox1.BorderStyle = System.Windows.Forms.BorderStyle.Fixed3D
        Me.PictureBox1.Location = New System.Drawing.Point(97, 211)
        Me.PictureBox1.Name = "PictureBox1"
        Me.PictureBox1.Size = New System.Drawing.Size(1239, 374)
        Me.PictureBox1.TabIndex = 0
        Me.PictureBox1.TabStop = False
        '
        'btnSaveLabel
        '
        Me.btnSaveLabel.Location = New System.Drawing.Point(1270, 593)
        Me.btnSaveLabel.Name = "btnSaveLabel"
        Me.btnSaveLabel.RightToLeft = System.Windows.Forms.RightToLeft.Yes
        Me.btnSaveLabel.Size = New System.Drawing.Size(66, 22)
        Me.btnSaveLabel.TabIndex = 1
        Me.btnSaveLabel.Text = "Save"
        Me.btnSaveLabel.UseVisualStyleBackColor = True
        '
        'btnOpenFolder
        '
        Me.btnOpenFolder.Location = New System.Drawing.Point(97, 591)
        Me.btnOpenFolder.Name = "btnOpenFolder"
        Me.btnOpenFolder.Size = New System.Drawing.Size(137, 26)
        Me.btnOpenFolder.TabIndex = 2
        Me.btnOpenFolder.Text = "Select Folder"
        Me.btnOpenFolder.UseVisualStyleBackColor = True
        '
        'txtLabel
        '
        Me.txtLabel.Location = New System.Drawing.Point(697, 593)
        Me.txtLabel.Name = "txtLabel"
        Me.txtLabel.Size = New System.Drawing.Size(472, 20)
        Me.txtLabel.TabIndex = 3
        '
        'btnPrev
        '
        Me.btnPrev.Image = CType(resources.GetObject("btnPrev.Image"), System.Drawing.Image)
        Me.btnPrev.Location = New System.Drawing.Point(11, 505)
        Me.btnPrev.Name = "btnPrev"
        Me.btnPrev.Size = New System.Drawing.Size(80, 80)
        Me.btnPrev.TabIndex = 4
        Me.btnPrev.Text = "Prev"
        Me.btnPrev.UseVisualStyleBackColor = True
        '
        'btnNext
        '
        Me.btnNext.Image = CType(resources.GetObject("btnNext.Image"), System.Drawing.Image)
        Me.btnNext.Location = New System.Drawing.Point(1342, 505)
        Me.btnNext.Name = "btnNext"
        Me.btnNext.Size = New System.Drawing.Size(80, 80)
        Me.btnNext.TabIndex = 5
        Me.btnNext.Text = "Next"
        Me.btnNext.UseVisualStyleBackColor = True
        '
        'lblImageInfo
        '
        Me.lblImageInfo.AutoSize = True
        Me.lblImageInfo.Location = New System.Drawing.Point(240, 596)
        Me.lblImageInfo.Name = "lblImageInfo"
        Me.lblImageInfo.Size = New System.Drawing.Size(39, 13)
        Me.lblImageInfo.TabIndex = 6
        Me.lblImageInfo.Text = "Label1"
        '
        'btnDelete
        '
        Me.btnDelete.Location = New System.Drawing.Point(1175, 593)
        Me.btnDelete.Name = "btnDelete"
        Me.btnDelete.Size = New System.Drawing.Size(75, 23)
        Me.btnDelete.TabIndex = 7
        Me.btnDelete.Text = "Delete"
        Me.btnDelete.UseVisualStyleBackColor = True
        '
        'Label1
        '
        Me.Label1.AutoSize = True
        Me.Label1.Location = New System.Drawing.Point(658, 598)
        Me.Label1.Name = "Label1"
        Me.Label1.Size = New System.Drawing.Size(33, 13)
        Me.Label1.TabIndex = 8
        Me.Label1.Text = "Label"
        '
        'PictureBox2
        '
        Me.PictureBox2.Location = New System.Drawing.Point(97, 12)
        Me.PictureBox2.Name = "PictureBox2"
        Me.PictureBox2.Size = New System.Drawing.Size(1239, 175)
        Me.PictureBox2.TabIndex = 9
        Me.PictureBox2.TabStop = False
        '
        'btnSkipLeft
        '
        Me.btnSkipLeft.Image = CType(resources.GetObject("btnSkipLeft.Image"), System.Drawing.Image)
        Me.btnSkipLeft.Location = New System.Drawing.Point(11, 211)
        Me.btnSkipLeft.Name = "btnSkipLeft"
        Me.btnSkipLeft.Size = New System.Drawing.Size(80, 80)
        Me.btnSkipLeft.TabIndex = 10
        Me.btnSkipLeft.Text = "Prev"
        Me.btnSkipLeft.UseVisualStyleBackColor = True
        '
        'btnSkipRight
        '
        Me.btnSkipRight.Image = CType(resources.GetObject("btnSkipRight.Image"), System.Drawing.Image)
        Me.btnSkipRight.Location = New System.Drawing.Point(1342, 211)
        Me.btnSkipRight.Name = "btnSkipRight"
        Me.btnSkipRight.Size = New System.Drawing.Size(80, 80)
        Me.btnSkipRight.TabIndex = 11
        Me.btnSkipRight.Text = "Next"
        Me.btnSkipRight.UseVisualStyleBackColor = True
        '
        'btnPathImage
        '
        Me.btnPathImage.Location = New System.Drawing.Point(14, 60)
        Me.btnPathImage.Name = "btnPathImage"
        Me.btnPathImage.Size = New System.Drawing.Size(54, 50)
        Me.btnPathImage.TabIndex = 12
        Me.btnPathImage.Text = "Button1"
        Me.btnPathImage.UseVisualStyleBackColor = True
        '
        'OpenFileDialog1
        '
        Me.OpenFileDialog1.FileName = "OpenFileDialog1"
        '
        'Form1
        '
        Me.AutoScaleDimensions = New System.Drawing.SizeF(6.0!, 13.0!)
        Me.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font
        Me.ClientSize = New System.Drawing.Size(1515, 633)
        Me.Controls.Add(Me.btnPathImage)
        Me.Controls.Add(Me.btnSkipRight)
        Me.Controls.Add(Me.btnSkipLeft)
        Me.Controls.Add(Me.PictureBox2)
        Me.Controls.Add(Me.Label1)
        Me.Controls.Add(Me.btnDelete)
        Me.Controls.Add(Me.lblImageInfo)
        Me.Controls.Add(Me.btnNext)
        Me.Controls.Add(Me.btnPrev)
        Me.Controls.Add(Me.txtLabel)
        Me.Controls.Add(Me.btnOpenFolder)
        Me.Controls.Add(Me.btnSaveLabel)
        Me.Controls.Add(Me.PictureBox1)
        Me.Name = "Form1"
        Me.Text = "Form1"
        CType(Me.PictureBox1, System.ComponentModel.ISupportInitialize).EndInit()
        CType(Me.PictureBox2, System.ComponentModel.ISupportInitialize).EndInit()
        Me.ResumeLayout(False)
        Me.PerformLayout()

    End Sub

    Friend WithEvents PictureBox1 As PictureBox
    Friend WithEvents btnSaveLabel As Button
    Friend WithEvents btnOpenFolder As Button
    Friend WithEvents txtLabel As TextBox
    Friend WithEvents btnPrev As Button
    Friend WithEvents btnNext As Button
    Friend WithEvents FolderBrowserDialog1 As FolderBrowserDialog
    Friend WithEvents lblImageInfo As Label
    Friend WithEvents btnDelete As Button
    Friend WithEvents Label1 As Label
    Friend WithEvents PictureBox2 As PictureBox
    Friend WithEvents btnSkipLeft As Button
    Friend WithEvents btnSkipRight As Button
    Friend WithEvents btnPathImage As Button
    Friend WithEvents OpenFileDialog1 As OpenFileDialog
End Class
