import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'DocScanner AI',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.teal, brightness: Brightness.dark),
        useMaterial3: true,
      ),
      home: const ScannerHomePage(),
    );
  }
}

class ScannerHomePage extends StatefulWidget {
  const ScannerHomePage({super.key});

  @override
  State<ScannerHomePage> createState() => _ScannerHomePageState();
}

class _ScannerHomePageState extends State<ScannerHomePage> with SingleTickerProviderStateMixin {
  File? _imageFile;
  File? _processedImageFile;
  bool _isProcessing = false;
  late AnimationController _animationController;
  final ImagePicker _picker = ImagePicker();

  @override
  void initState() {
    super.initState();
    _animationController = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 2),
    );
  }

  @override
  void dispose() {
    _animationController.dispose();
    super.dispose();
  }

  Future<void> _pickImage(ImageSource source) async {
    try {
      final pickedFile = await _picker.pickImage(source: source);
      if (pickedFile != null) {
        setState(() {
          _imageFile = File(pickedFile.path);
          _processedImageFile = null;
        });
        _uploadAndProcessImage();
      }
    } catch (e) {
      _showErrorSnackBar('Error picking image: $e');
    }
  }

  Future<void> _uploadAndProcessImage() async {
    if (_imageFile == null) return;

    setState(() {
      _isProcessing = true;
      _processedImageFile = null;
    });
    _animationController.repeat(reverse: true);

    try {
      // Xác định IP truy cập backend tuỳ server
      final baseUrl = Platform.isAndroid ? 'http://10.0.2.2:8000' : 'http://127.0.0.1:8000';
      final uri = Uri.parse('$baseUrl/api/scan');
      
      var request = http.MultipartRequest('POST', uri);
      request.files.add(await http.MultipartFile.fromPath('image', _imageFile!.path));

      var response = await request.send();

      if (response.statusCode == 200) {
        var bytes = await response.stream.toBytes();
        final dir = await getTemporaryDirectory();
        final processedFile = File('${dir.path}/processed_${DateTime.now().millisecondsSinceEpoch}.jpg');
        await processedFile.writeAsBytes(bytes);

        setState(() {
          _processedImageFile = processedFile;
        });
      } else {
        _showErrorSnackBar('Server Error: Lỗi xử lý ảnh từ Backend (${response.statusCode})');
      }
    } catch (e) {
      _showErrorSnackBar('Lỗi kết nối Server: $e');
    } finally {
      setState(() {
        _isProcessing = false;
      });
      _animationController.stop();
      _animationController.reset();
    }
  }

  void _showErrorSnackBar(String message) {
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(
      content: Text(message),
      backgroundColor: Colors.redAccent,
    ));
  }

  Widget _buildImagePreview() {
    if (_processedImageFile != null) {
      // Hiện ảnh đã quét
      return Stack(
        fit: StackFit.expand,
        children: [
          Image.file(_processedImageFile!, fit: BoxFit.contain),
          Positioned(
            top: 10,
            right: 10,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
              decoration: BoxDecoration(
                color: Colors.green.shade800,
                borderRadius: BorderRadius.circular(20),
              ),
              child: const Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(Icons.check_circle, color: Colors.white, size: 16),
                  SizedBox(width: 6),
                  Text("Quét AI Hoàn Tất", style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
                ],
              ),
            ),
          )
        ],
      );
    } else if (_imageFile != null) {
      // Hiện ảnh gốc với overlay xử lý
      return Stack(
        fit: StackFit.expand,
        children: [
          Image.file(_imageFile!, fit: BoxFit.contain),
          if (_isProcessing)
            AnimatedBuilder(
              animation: _animationController,
              builder: (context, child) {
                return CustomPaint(
                  painter: ScannerPainter(_animationController.value),
                );
              },
            ),
          if (_isProcessing)
            Container(color: Colors.black45), // Làm tối màn hình mờ đi xíu
          if (_isProcessing)
            const Center(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                   CircularProgressIndicator(
                    valueColor: AlwaysStoppedAnimation<Color>(Colors.greenAccent),
                  ),
                  SizedBox(height: 16),
                  Text("Đang phân tích viền tài liệu...", 
                    style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold, fontSize: 16)
                  ),
                ],
              ),
            ),
        ],
      );
    } else {
      return const Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.document_scanner, size: 80, color: Colors.teal),
            SizedBox(height: 16),
            Text(
              'Chưa có ảnh tải lên\nVui lòng chụp hoặc chọn ảnh',
              textAlign: TextAlign.center,
              style: TextStyle(fontSize: 18, color: Colors.grey),
            ),
          ],
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        title: const Text('DocScanner ML', style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
        backgroundColor: Colors.teal.shade900,
        centerTitle: true,
        elevation: 0,
      ),
      body: SafeArea(
        child: Column(
          children: [
            Expanded(
              child: Container(
                margin: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Colors.grey.shade900,
                  borderRadius: BorderRadius.circular(16),
                  border: Border.all(color: Colors.teal.shade700, width: 2),
                  boxShadow: [
                    BoxShadow(color: Colors.teal.withValues(alpha: 0.1), blurRadius: 10, spreadRadius: 2),
                  ]
                ),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(14),
                  child: _buildImagePreview(),
                ),
              ),
            ),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 20),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: [
                  Expanded(
                    child: ElevatedButton.icon(
                      onPressed: _isProcessing ? null : () => _pickImage(ImageSource.gallery),
                      icon: const Icon(Icons.photo_library),
                      label: const Text('Thư Viện'),
                      style: ElevatedButton.styleFrom(
                        padding: const EdgeInsets.symmetric(vertical: 16),
                        backgroundColor: Colors.teal.shade800,
                        foregroundColor: Colors.white,
                        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                      ),
                    ),
                  ),
                  const SizedBox(width: 16),
                  Expanded(
                    child: ElevatedButton.icon(
                      onPressed: _isProcessing ? null : () => _pickImage(ImageSource.camera),
                      icon: const Icon(Icons.camera_alt),
                      label: const Text('Trực Tiếp'),
                      style: ElevatedButton.styleFrom(
                        padding: const EdgeInsets.symmetric(vertical: 16),
                        backgroundColor: Colors.teal.shade600,
                        foregroundColor: Colors.white,
                        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class ScannerPainter extends CustomPainter {
  final double animationValue;

  ScannerPainter(this.animationValue);

  @override
  void paint(Canvas canvas, Size size) {
    final lineY = size.height * animationValue;
    final paint = Paint()
      ..color = Colors.greenAccent
      ..strokeWidth = 3.0
      ..style = PaintingStyle.stroke;
      
    final shadowPaint = Paint()
      ..shader = LinearGradient(
        begin: Alignment.topCenter,
        end: Alignment.bottomCenter,
        colors: [
          Colors.greenAccent.withValues(alpha: 0.0),
          Colors.greenAccent.withValues(alpha: 0.4),
        ],
      ).createShader(Rect.fromLTWH(0, lineY - 80, size.width, 80));

    // Draw the gradient tail
    canvas.drawRect(Rect.fromLTWH(0, lineY - 80, size.width, 80), shadowPaint);
    
    // Draw the scanning line
    canvas.drawLine(Offset(0, lineY), Offset(size.width, lineY), paint);
  }

  @override
  bool shouldRepaint(covariant ScannerPainter oldDelegate) {
    return oldDelegate.animationValue != animationValue;
  }
}
