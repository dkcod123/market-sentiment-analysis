import { Component, Input, OnInit, OnChanges, SimpleChanges } from '@angular/core';
import { ChartConfiguration, ChartType } from 'chart.js';

@Component({
  selector: 'app-trend-graph',
  templateUrl: './trend-graph.component.html',
  styleUrls: ['./trend-graph.component.scss']
})
export class TrendGraphComponent implements OnInit, OnChanges {
  @Input() stock: string = '';
  @Input() timeframe: string = '7d';
  @Input() trendData: { labels: string[], data: number[] } | null = null;

  public lineChartData: ChartConfiguration<'line'>['data'] = {
    labels: [],
    datasets: [
      {
        data: [],
        label: 'Sentiment',
        fill: false,
        borderColor: '#3f51b5',
        tension: 0.3
      }
    ]
  };
  public lineChartType: ChartType = 'line';

  constructor() {}

  ngOnInit(): void {
    this.updateChart();
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['trendData']) {
      this.updateChart();
    }
  }

  updateChart() {
    if (this.trendData) {
      this.lineChartData.labels = this.trendData.labels;
      this.lineChartData.datasets[0].data = this.trendData.data;
    } else {
      this.lineChartData.labels = [];
      this.lineChartData.datasets[0].data = [];
    }
  }
} 