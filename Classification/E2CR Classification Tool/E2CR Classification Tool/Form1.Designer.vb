<Global.Microsoft.VisualBasic.CompilerServices.DesignerGenerated()>
Partial Class Form1
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
        Dim resources As System.ComponentModel.ComponentResourceManager = New System.ComponentModel.ComponentResourceManager(GetType(Form1))
        ToolStrip1 = New ToolStrip()
        btnSelectFolder = New ToolStripButton()
        Panel1 = New Panel()
        pbImagesToLabel = New PictureBox()
        HScrollBar1 = New HScrollBar()
        GroupBox1 = New GroupBox()
        pbRow = New PictureBox()
        GroupBox2 = New GroupBox()
        pbRowCS = New PictureBox()
        txtImageName = New TextBox()
        Label1 = New Label()
        txtLabel = New TextBox()
        ToolStrip1.SuspendLayout()
        Panel1.SuspendLayout()
        CType(pbImagesToLabel, ComponentModel.ISupportInitialize).BeginInit()
        GroupBox1.SuspendLayout()
        CType(pbRow, ComponentModel.ISupportInitialize).BeginInit()
        GroupBox2.SuspendLayout()
        CType(pbRowCS, ComponentModel.ISupportInitialize).BeginInit()
        SuspendLayout()
        ' 
        ' ToolStrip1
        ' 
        ToolStrip1.ImageScalingSize = New Size(20, 20)
        ToolStrip1.Items.AddRange(New ToolStripItem() {btnSelectFolder})
        ToolStrip1.Location = New Point(0, 0)
        ToolStrip1.Name = "ToolStrip1"
        ToolStrip1.Size = New Size(1176, 27)
        ToolStrip1.TabIndex = 2
        ToolStrip1.Text = "ToolStrip1"
        ' 
        ' btnSelectFolder
        ' 
        btnSelectFolder.DisplayStyle = ToolStripItemDisplayStyle.Image
        btnSelectFolder.Image = CType(resources.GetObject("btnSelectFolder.Image"), Image)
        btnSelectFolder.ImageTransparentColor = Color.Magenta
        btnSelectFolder.Name = "btnSelectFolder"
        btnSelectFolder.Size = New Size(29, 24)
        btnSelectFolder.Text = "ToolStripButton1"
        ' 
        ' Panel1
        ' 
        Panel1.AutoScroll = True
        Panel1.Controls.Add(pbImagesToLabel)
        Panel1.Location = New Point(0, 439)
        Panel1.Name = "Panel1"
        Panel1.Size = New Size(1176, 162)
        Panel1.TabIndex = 3
        ' 
        ' pbImagesToLabel
        ' 
        pbImagesToLabel.Location = New Point(3, -48)
        pbImagesToLabel.Name = "pbImagesToLabel"
        pbImagesToLabel.Size = New Size(1170, 208)
        pbImagesToLabel.TabIndex = 0
        pbImagesToLabel.TabStop = False
        ' 
        ' HScrollBar1
        ' 
        HScrollBar1.Location = New Point(3, 602)
        HScrollBar1.Name = "HScrollBar1"
        HScrollBar1.Size = New Size(1170, 26)
        HScrollBar1.TabIndex = 1
        ' 
        ' GroupBox1
        ' 
        GroupBox1.Controls.Add(pbRow)
        GroupBox1.Location = New Point(3, 47)
        GroupBox1.Name = "GroupBox1"
        GroupBox1.Size = New Size(1170, 166)
        GroupBox1.TabIndex = 4
        GroupBox1.TabStop = False
        GroupBox1.Text = "GroupBox1"
        ' 
        ' pbRow
        ' 
        pbRow.Location = New Point(6, 26)
        pbRow.Name = "pbRow"
        pbRow.Size = New Size(1158, 135)
        pbRow.TabIndex = 0
        pbRow.TabStop = False
        ' 
        ' GroupBox2
        ' 
        GroupBox2.Controls.Add(pbRowCS)
        GroupBox2.Location = New Point(3, 219)
        GroupBox2.Name = "GroupBox2"
        GroupBox2.Size = New Size(1170, 166)
        GroupBox2.TabIndex = 5
        GroupBox2.TabStop = False
        GroupBox2.Text = "GroupBox2"
        ' 
        ' pbRowCS
        ' 
        pbRowCS.Location = New Point(6, 26)
        pbRowCS.Name = "pbRowCS"
        pbRowCS.Size = New Size(1158, 135)
        pbRowCS.TabIndex = 0
        pbRowCS.TabStop = False
        ' 
        ' txtImageName
        ' 
        txtImageName.Enabled = False
        txtImageName.Location = New Point(3, 631)
        txtImageName.Name = "txtImageName"
        txtImageName.Size = New Size(1170, 27)
        txtImageName.TabIndex = 6
        txtImageName.TextAlign = HorizontalAlignment.Center
        ' 
        ' Label1
        ' 
        Label1.AutoSize = True
        Label1.Font = New Font("Segoe UI", 18F, FontStyle.Bold, GraphicsUnit.Point, CByte(0))
        Label1.Location = New Point(311, 664)
        Label1.Name = "Label1"
        Label1.Size = New Size(101, 41)
        Label1.TabIndex = 7
        Label1.Text = "Label:"
        ' 
        ' txtLabel
        ' 
        txtLabel.Enabled = False
        txtLabel.Font = New Font("Segoe UI", 18F, FontStyle.Regular, GraphicsUnit.Point, CByte(0))
        txtLabel.Location = New Point(418, 664)
        txtLabel.Name = "txtLabel"
        txtLabel.Size = New Size(398, 47)
        txtLabel.TabIndex = 8
        ' 
        ' Form1
        ' 
        AutoScaleDimensions = New SizeF(8F, 20F)
        AutoScaleMode = AutoScaleMode.Font
        ClientSize = New Size(1176, 754)
        Controls.Add(HScrollBar1)
        Controls.Add(txtLabel)
        Controls.Add(Label1)
        Controls.Add(txtImageName)
        Controls.Add(GroupBox2)
        Controls.Add(GroupBox1)
        Controls.Add(Panel1)
        Controls.Add(ToolStrip1)
        Name = "Form1"
        Text = "Form1"
        ToolStrip1.ResumeLayout(False)
        ToolStrip1.PerformLayout()
        Panel1.ResumeLayout(False)
        CType(pbImagesToLabel, ComponentModel.ISupportInitialize).EndInit()
        GroupBox1.ResumeLayout(False)
        CType(pbRow, ComponentModel.ISupportInitialize).EndInit()
        GroupBox2.ResumeLayout(False)
        CType(pbRowCS, ComponentModel.ISupportInitialize).EndInit()
        ResumeLayout(False)
        PerformLayout()
    End Sub
    Friend WithEvents ToolStrip1 As ToolStrip
    Friend WithEvents btnSelectFolder As ToolStripButton
    Friend WithEvents Panel1 As Panel
    Friend WithEvents pbImagesToLabel As PictureBox
    Friend WithEvents HScrollBar1 As HScrollBar
    Friend WithEvents GroupBox1 As GroupBox
    Friend WithEvents pbRow As PictureBox
    Friend WithEvents GroupBox2 As GroupBox
    Friend WithEvents pbRowCS As PictureBox
    Friend WithEvents txtImageName As TextBox
    Friend WithEvents Label1 As Label
    Friend WithEvents txtLabel As TextBox

End Class
