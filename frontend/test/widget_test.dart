import 'package:flutter_test/flutter_test.dart';
import 'package:autoscore_nidra/src/app.dart';

void main() {
  testWidgets('renders the sleep EEG viewer shell', (tester) async {
    await tester.pumpWidget(const ScoringNidraApp());

    expect(find.text('Jump to epoch:'), findsOneWidget);
    expect(find.textContaining('Ready'), findsOneWidget);
  });
}
