import { Component, OnInit } from '@angular/core';
import { SentimentApiService } from '../../services/sentiment-api.service';

@Component({
  selector: 'app-stock-table',
  templateUrl: './stock-table.component.html',
  styleUrls: ['./stock-table.component.scss']
})
export class StockTableComponent implements OnInit {
  displayedColumns: string[] = ['stock', 'avg_sentiment', 'total_weight', 'twitter_count', 'reddit_count'];
  dataSource: any[] = [];
  loading = true;

  constructor(private sentimentApi: SentimentApiService) {}

  ngOnInit(): void {
    this.sentimentApi.getAllStocksSentiment('24h').subscribe({
      next: (res) => {
        this.dataSource = res.results || [];
        this.loading = false;
      },
      error: (err) => {
        this.loading = false;
        // handle error
      }
    });
  }
} 