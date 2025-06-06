package com.baidu.paddle.lite.demo.ppocr_demo;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.TableLayout;
import android.widget.TableRow;
import android.widget.TextView;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;


public class TableActivity extends Activity {
    private static final String TAG = "TableActivity";

    @SuppressLint("StaticFieldLeak")
    public static TableLayout resultsTable;
    ArrayList<String> ocr_res;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.table_view);
        Button returnButton = (Button) findViewById(R.id.button);
        resultsTable = findViewById(R.id.results_table);

        returnButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(TableActivity.this, MainActivity.class);
                startActivity(intent);
            }
        });
    }

    @Override
    protected void onResume() {
        super.onResume();

        ocr_res = getIntent().getStringArrayListExtra("list");

        assert ocr_res != null;
        for (String result : ocr_res) {
            addTableRow(result, "1"); // Replace with actual quantity detection
        }


    }

    private void addTableRow(String item, String quantity) {
        TableRow row = new TableRow(this);

        TextView itemView = new TextView(this);
        itemView.setText(item);
        itemView.setPadding(16, 8, 16, 8);
        itemView.setTextDirection(View.TEXT_DIRECTION_ANY_RTL);

        TextView qtyView = new TextView(this);
        qtyView.setText(quantity);
        qtyView.setPadding(16, 8, 16, 8);
        qtyView.setTextAlignment(View.TEXT_ALIGNMENT_VIEW_END);

        row.addView(itemView);
        row.addView(qtyView);
        resultsTable.addView(row);
    }
}