import { Component, OnInit, OnDestroy } from '@angular/core';
import { SentimentApiService } from '../services/sentiment-api.service';
import { Subscription, timer } from 'rxjs';

@Component({
  selector: 'app-dashboard',
  templateUrl: './dashboard.component.html',
  styleUrls: ['./dashboard.component.scss']
})
export class DashboardComponent implements OnInit, OnDestroy {
  selectedTimeframe = '24h';
  selectedSentiment = '';
  selectedStock = '';
  allStocks: any[] = [];
  filteredStocks: any[] = [];
  trendData: { labels: string[], data: number[] } = { labels: [], data: [] };
  lastUpdated: Date | null = null;
  private pollSub: Subscription | null = null;
  readonly POLL_INTERVAL = 5 * 60 * 1000; // 5 minutes
  alertMessages: string[] = [];
  private previousSentiments: { [stock: string]: number } = {};

  constructor(private sentimentApi: SentimentApiService) {}

  ngOnInit(): void {
    this.startPolling();
  }

  ngOnDestroy(): void {
    this.pollSub?.unsubscribe();
  }

  startPolling() {
    this.pollSub?.unsubscribe();
    this.pollSub = timer(0, this.POLL_INTERVAL).subscribe(() => {
      this.fetchAllStocks();
      this.updateTrendData();
      this.lastUpdated = new Date();
    });
  }

  manualRefresh() {
    this.fetchAllStocks();
    this.updateTrendData();
    this.lastUpdated = new Date();
  }

  fetchAllStocks() {
    this.sentimentApi.getAllStocksSentiment(this.selectedTimeframe).subscribe({
      next: (res) => {
        this.allStocks = res.results || [];
        this.checkForSentimentSpikes();
        this.applyFilters();
      }
    });
  }

  checkForSentimentSpikes() {
    this.alertMessages = [];
    for (const stock of this.allStocks) {
      const prev = this.previousSentiments[stock.stock] ?? stock.avg_sentiment;
      const curr = stock.avg_sentiment;
      if (Math.abs(curr - prev) > 0.5) {
        this.alertMessages.push(
          `Alert: Sudden sentiment spike for ${stock.stock} (Δ=${(curr - prev).toFixed(2)})`
        );
      }
      this.previousSentiments[stock.stock] = curr;
    }
  }

  onTimeframeChange(tf: string) {
    this.selectedTimeframe = tf;
    this.manualRefresh();
  }
  onSentimentChange(sent: string) {
    this.selectedSentiment = sent;
    this.applyFilters();
  }
  onStockChange(stock: string) {
    this.selectedStock = stock;
    this.applyFilters();
    this.updateTrendData();
  }

  applyFilters() {
    this.filteredStocks = this.allStocks.filter(stock => {
      const sentimentMatch = this.selectedSentiment ? (stock.sentiment_label === this.selectedSentiment) : true;
      const stockMatch = this.selectedStock ? (stock.stock === this.selectedStock) : true;
      return sentimentMatch && stockMatch;
    });
  }

  updateTrendData() {
    if (!this.selectedStock) {
      this.trendData = { labels: [], data: [] };
      return;
    }
    this.sentimentApi.getStockSentiment(this.selectedStock, this.selectedTimeframe).subscribe({
      next: (res) => {
        this.trendData = {
          labels: [this.selectedTimeframe],
          data: [res.avg_sentiment]
        };
      }
    });
  }
} 